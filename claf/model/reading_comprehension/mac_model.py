import torch
from torch.autograd import Variable
from torch import nn
from torch.nn.init import kaiming_uniform_, xavier_uniform_, normal
import torch.nn.functional as F
from claf.model.reading_comprehension.biatt import BiAttention, LockedDropout

attns = {
    "read": None,
    "control": None
}

class CtrlBiAttention(nn.Module):
    def __init__(self, input_size, dropout):
        super().__init__()
        self.dropout = LockedDropout(dropout)
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)

        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))

    def forward(self, input, memory, mask, ctrl_attn):
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input = self.dropout(input)
        memory = self.dropout(memory)

        input_dot = self.input_linear(input)
        memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
        cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
        att = input_dot + memory_dot + cross_dot
        att = att - 1e30 * (1 - mask[:,None])

        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        weight_two = F.softmax(torch.bmm(att, ctrl_attn).squeeze(), dim=-1).view(bsz, 1, input_len)
        output_two = torch.bmm(weight_two, input)

        return torch.cat([input, output_one, input*output_one, output_two*output_one], dim=-1)


def linear(in_dim, out_dim, bias=True):
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()
    return lin


class ControlUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.shared_control_proj = linear(self.dim, self.dim)
        self.position_aware = nn.ModuleList()
        for i in range(2):
            self.position_aware.append(linear(self.dim, self.dim))    # if controlInputUnshared

        self.control_question = linear(self.dim * 2, self.dim)
        self.attn = linear(self.dim, 1)


    def forward(self, context, question, controls, question_mask):
        cur_step = len(controls) - 1
        control = controls[-1]

        question = torch.tanh(self.shared_control_proj(question))       # TODO: avoid repeating call
        position_aware = self.position_aware[cur_step](question)

        control_question = torch.cat([control, position_aware], 1)
        control_question = self.control_question(control_question)
        control_question = control_question.unsqueeze(1)

        context_prod = control_question * context

        # ++ optionally concatenate words (= context)

        # optional projection (if config.controlProj) --> stacks another linear after activation

        attn_weight = self.attn(context_prod).squeeze() - 1e30 * (1 - question_mask)

        attn = F.softmax(attn_weight, 1).unsqueeze(2)

        attns["control"].append(attn.squeeze())

        # only valid if self.inwords == self.outwords
        next_control = (attn * context).sum(1)

        return next_control, attn


class ReadUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.read_dropout = nn.Dropout(0.15)
        self.mem_proj = linear(dim, dim)
        self.kb_proj = linear(dim, dim)
        self.concat = linear(dim * 2, dim)
        self.concat2 = linear(dim, dim)
        self.attn = linear(dim, 1)
        self.lin = linear(dim*4, dim)
        self.bi = CtrlBiAttention(dim, 0)

    def forward(self, memory, know, control, masks, document_mask, question, question_mask, ctrl_attn):
        proj_mem = self.mem_proj(memory[-1]).unsqueeze(1)
        proj_know = self.kb_proj(know)
        concat = self.concat(torch.cat([
            proj_mem * proj_know, 
            proj_know       
        ], 2))

        ## Step 2: compute interactions with control (if config.readCtrl)
        out = F.elu(self.lin(self.bi(concat, question, question_mask, ctrl_attn)))

        # if readCtrlConcatInter torch.cat([interactions, concat])

        # optionally concatenate knowledge base elements

        # optional nonlinearity

        attn = self.read_dropout(out)
        attn = self.attn(attn).squeeze() - 1e30 * (1 - document_mask)
        attn = F.softmax(attn, 1).unsqueeze(2)

        attns["read"].append(attn.squeeze())

        read = (attn * know).sum(1)

        return read, out


class WriteUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.self_attention = False
        self.merge_control = True
        concat_dim = 2

        if self.self_attention:
            self.control = linear(dim, dim)
            self.attn = linear(dim, 1)
            concat_dim += 1

        if self.merge_control:
            concat_dim += 1
        self.concat = linear(dim * concat_dim, dim)

    def forward(self, memories, retrieved, question, controls):
        # optionally project info if config.writeInfoProj:

        # optional info nonlinearity if writeInfoAct != 'NON'

        # compute self-attention vector based on previous controls and memories
        if self.self_attention:
            selfControl = controls[-1]
            selfControl = self.control(selfControl)
            controls_cat = torch.stack(controls[:-1], 2)
            attn = selfControl.unsqueeze(2) * controls_cat
            attn = self.attn(attn.permute(0, 2, 1))
            attn = F.softmax(attn, 1).permute(0, 2, 1)

            memories_cat = torch.stack(memories, 2)
            attn_mem = (attn * memories_cat).sum(2)
            # next_mem = self.W_s(attn_mem) + self.W_p(concat)


        prev_mem = memories[-1]
        # get write unit inputs: previous memory, the new info, optionally self-attention / control
        concat = torch.cat([retrieved, prev_mem], 1)

        if self.self_attention:
            concat = torch.cat([concat, attn_mem], 1)

        # optionally merge current control state into memory. (writeMergeCtrl)
        if self.merge_control:
            concat = torch.cat([concat, controls[-1]], 1)

        # project memory back to memory dimension if config.writeMemProj
        concat = self.concat(concat)

        # optional memory nonlinearity (not implemented)

        # write unit gate moved to RNNWrapper

        next_mem = concat

        return next_mem


class MACCell(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.control = ControlUnit(dim)
        self.read = ReadUnit(dim)
        self.write = WriteUnit(dim)

        self.mem_0 = nn.Parameter(torch.zeros(1, dim))
        # control0 is most often question, other times (eg. args2.txt) its a learned parameter initialized as random normal

        self.dim = dim
    
    def get_mask(self, x, dropout):
        mask = torch.empty_like(x).bernoulli_(1 - dropout)
        mask = mask / (1 - dropout)
        return mask

    def init_hidden(self, b_size, question):
        if True:#self.cfg.MAC.INIT_CNTRL_AS_Q:
            control = question
        else:
            control = self.control_0.expand(b_size, self.dim)
        memory = self.mem_0.expand(b_size, self.dim)
        if self.training: #and self.cfg.MAC.MEMORY_VAR_DROPOUT:
            memory_mask = self.get_mask(memory, 0.1)
        else:
            memory_mask = None

        controls = [control]
        memories = [memory]

        return (controls, memories), (memory_mask)

    def forward(self, inputs, state, masks):
        words, question, img, question_mask, document_mask = inputs
        controls, memories = state

        control, ctrl_attn = self.control(words, question, controls, question_mask)
        controls.append(control)

        read, out = self.read(memories, img, controls, masks, document_mask, words, question_mask, ctrl_attn)
        # if config.writeDropout < 1.0:     dropouts["write"]
        memory = self.write(memories, read, question, controls)
        memories.append(memory)

        return (controls, memories), out


class OutputUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.question_proj = nn.Linear(dim, dim)
        self.classifier_out = nn.Sequential(nn.Dropout(p=0.15),       # output dropout outputDropout=0.85
                                        nn.Linear(dim * 2, dim),
                                        nn.ELU(),
                                        nn.Dropout(p=0.15),       # output dropout outputDropout=0.85
                                        nn.Linear(dim, dim))
        xavier_uniform_(self.classifier_out[1].weight)
        xavier_uniform_(self.classifier_out[4].weight)
    
    def forward(self, last_mem, question):
        question = self.question_proj(question)
        cat = torch.cat([last_mem, question], 1)
        out = self.classifier_out(cat)
        return out


class RecurrentWrapper(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.controller = MACCell(dim)

        self.dim = dim
        self.gate = linear(dim, 1)
    
    def forward(self, *inputs):
        state, masks = self.controller.init_hidden(inputs[1].size(0), inputs[1])
        outs = []

        for _ in range(1, 3):
            state, out = self.controller(inputs, state, masks)
            outs.append(out)
            # memory gate
            if True:
                controls, memories = state
                gate = torch.sigmoid(self.gate(controls[-1]) + 1.0)
                memories[-1] = gate * memories[-2] + (1 - gate) * memories[-1]
        
        _, memories = state
        return memories, outs


class MACNetwork(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.question_dropout = nn.Dropout(0.1)
        hDim = int(dim / 2)
        self.lstm = nn.LSTM(dim, hDim,
                        # dropout=cfg.MAC.ENC_INPUT_DROPOUT,
                        batch_first=True, bidirectional=True)

        # choose different wrappers for no-act/actSmooth/actBaseline
        self.actmac = RecurrentWrapper(dim)
        self.dim = dim

    def forward(self, context, question, question_len, question_mask, document_mask, dropout=0.15):
        b_size = question.size(0)
        q_words = question
        attns["read"] = []
        attns["control"] = []

        embed = nn.utils.rnn.pack_padded_sequence(question, question_len, batch_first=True, enforce_sorted=False)
        lstm_out, (h, _) = self.lstm(embed)
        question = torch.cat([h[0], h[1]], -1)
        question = self.question_dropout(question)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        h = h.permute(1, 0, 2).contiguous().view(b_size, -1)

        memories, out = self.actmac(q_words, question, context, question_mask, document_mask)
        # return memories and encoded question
        return memories, question, out
