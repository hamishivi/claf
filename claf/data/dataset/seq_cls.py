
import json
from overrides import overrides
import torch

from claf.data import utils
from claf.data.collate import PadCollator
from claf.data.dataset.base import DatasetBase


class SeqClsDataset(DatasetBase):
    """
    Dataset for Sequence Classification

    * Args:
        batch: Batch DTO (claf.data.batch)

    * Kwargs:
        helper: helper from data_reader
    """

    def __init__(self, batch, helper=None):
        super(SeqClsDataset, self).__init__()

        self.name = "seq_cls"
        self.helper = helper
        self.raw_dataset = helper["raw_dataset"]

        self.class_idx2text = helper["class_idx2text"]

        self.sequences = {feature["id"]: feature["sequence"]["text"] for feature in batch.features}

        # Features
        self.sequence_idxs = [feature["sequence"] for feature in batch.features]

        self.features = [self.sequence_idxs]  # for lazy evaluation

        # Labels
        self.data_ids = {data_index: label["id"] for (data_index, label) in enumerate(batch.labels)}
        self.data_indices = list(self.data_ids.keys())

        self.classes = {
            label["id"]: {
                "class_idx": label["class_idx"],
                "class_text": label["class_text"],
            }
            for label in batch.labels
        }

        self.class_text = [label["class_text"] for label in batch.labels]
        self.class_idx = [label["class_idx"] for label in batch.labels]

    @overrides
    def collate_fn(self, cuda_device_id=None):
        """ collate: indexed features and labels -> tensor """
        collator = PadCollator(cuda_device_id=cuda_device_id)

        def make_tensor_fn(data):
            data_idxs, sequence_idxs, class_idxs = zip(*data)

            features = {
                "sequence": utils.transpose(sequence_idxs, skip_keys=["text"]),
            }
            labels = {
                "class_idx": class_idxs,
                "data_idx": data_idxs,
            }
            return collator(features, labels)

        return make_tensor_fn

    @overrides
    def __getitem__(self, index):
        self.lazy_evaluation(index)

        return (
            self.data_indices[index],
            self.sequence_idxs[index],
            self.class_idx[index],
        )

    def __len__(self):
        return len(self.data_ids)

    def __repr__(self):
        dataset_properties = {
            "name": self.name,
            "total_count": self.__len__(),
            "num_classes": self.num_classes,
            "sequence_maxlen": self.sequence_maxlen,
            "classes": self.class_idx2text,
        }
        return json.dumps(dataset_properties, indent=4)

    @property
    def num_classes(self):
        return len(self.class_idx2text)

    @property
    def sequence_maxlen(self):
        return self._get_feature_maxlen(self.sequence_idxs)

    def get_id(self, data_index):
        return self.data_ids[data_index]

    @overrides
    def get_ground_truth(self, data_id):
        return self.classes[data_id]

    def get_class_text_with_idx(self, class_index):
        if class_index is None:
            raise ValueError("class_index is required.")

        return self.class_idx2text[class_index]
