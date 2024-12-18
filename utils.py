#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch


class presDataset(torch.utils.data.Dataset):
    def __init__(self, a, b):
        self.pS_array, self.pH_array = a, b

    def __getitem__(self, idx):
        sid = self.pS_array[idx]
        hid = self.pH_array[idx]
        return sid, hid

    def __len__(self):
        return self.pH_array.shape[0]
