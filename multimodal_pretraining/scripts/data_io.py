import h5py
from torch.utils.data import Dataset, DataLoader
import re
import torchaudio
import torch
import torch.nn as nn


class MultimodalDataset(Dataset):
    def __init__(self, data_path):
        self.data = self.prep_data(data_path)

    def prep_data(self, data_path):
        data = {}
        with h5py.File(data_path, "r") as f:
            for k, v in f.items():
                if k == "size":
                    continue
                # for key, val in v.items():
                #     print(key, val)
                print(k, v.items(), v["Text"][:][0].decode(), v["Audio"][:].shape, v["Video"][:].shape)
                # raw_data[k] = [(v["ID"][i].decode(),
                #                 self.kw_spotting(v["Text"][i].decode()),
                #                 self.convert_label(v["Intention"][i].decode(), self.intention_label),
                #                 self.convert_label(v["Code"][i].decode(), self.activity_label),
                #                 self.extarct_feature(v["Audio"][i]))
                #                for i in range(len(v["ID"])) if
                #                v["Intention"][i].decode() in self.intention_label.keys()]
        return data


if __name__ == '__main__':
    dataset = MultimodalDataset(
        "/home/xixihahaggg/PycharmProjects/fairseq/multimodal_pretraining/data/val/data_set_000.h5py")
