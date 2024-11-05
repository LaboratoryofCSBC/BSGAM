#!/usr/bin/env python
# -*- coding:utf-8 -*-
from utils import *
from BSGAM import *
import sys
import os
import numpy as np
import torch
from torch_geometric.data import Data
import argparse
import datetime
import random
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=512)
parser.add_argument('--att_head_num', type=int, default=4)
parser.add_argument('--att_drop', type=float, default=0.0)
parser.add_argument('--BCE_L', type=float, default=0.025)
parser.add_argument('--seed', type=int, default=2023)
args = parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True


set_seed(args.seed)


class CustomBCEWithLogitsLoss(torch.nn.Module):
    def __init__(self):
        super(CustomBCEWithLogitsLoss, self).__init__()

    def forward(self, logits, targets, BCE_L):
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        probs = torch.sigmoid(logits)
        top20_probs, top20_indices = torch.topk(probs, 20, dim=1)
        top20_loss = 0
        for i in range(logits.size(0)):
            top20_targets = targets[i, top20_indices[i]]
            top20_loss += torch.nn.functional.binary_cross_entropy(top20_probs[i], top20_targets)
        top20_loss /= logits.size(0)
        total_loss = bce_loss + BCE_L * top20_loss
        return total_loss


class Logger(object):
    def __init__(self, filename=None):
        self.terminal = sys.stdout
        if filename is None:
            filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = Logger()

sh_edge = np.load('./data/sh_graph_train_1.npy').tolist()
sh_edge_index = torch.tensor(sh_edge, dtype=torch.long)
sh_x = torch.tensor([[i] for i in range(1201)], dtype=torch.float)
sh_data = Data(x=sh_x, edge_index=sh_edge_index.t().contiguous())

ss_edge = np.load('./data/ss_graph_train_3.npy').tolist()
ss_edge_index = torch.tensor(ss_edge, dtype=torch.long)
ss_x = torch.tensor([[i] for i in range(390)], dtype=torch.float)
ss_data = Data(x=ss_x, edge_index=ss_edge_index.t().contiguous())

hh_edge = np.load('./data/hh_graph_train_20.npy').tolist()
hh_edge_index = torch.tensor(hh_edge, dtype=torch.long) - 390
hh_x = torch.tensor([[i] for i in range(390, 1201)], dtype=torch.float)
hh_data = Data(x=hh_x, edge_index=hh_edge_index.t().contiguous())

prescript = pd.read_csv('./data/prescript.csv', encoding='utf-8')
pLen = len(prescript)
pS_array = np.zeros((pLen, 390))
pH_array = np.zeros((pLen, 811))

for i in range(pLen):
    j = eval(prescript.iloc[i, 0])
    pS_array[i, j] = 1
    k = eval(prescript.iloc[i, 1])
    k = [x - 390 for x in k]
    pH_array[i, k] = 1

pS_array = torch.from_numpy(pS_array).to(device).float()
pH_array = torch.from_numpy(pH_array).to(device).float()

sh_tensor = torch.load('./data/sh_tensor.pt').to(device)
s_tensor = torch.load('./data/s_tensor.pt').to(device)
h_tensor = torch.load('./data/h_tensor.pt').to(device)

kg_oneHot = np.load('./data/herb_oneHot.npy')
kg_oneHot = torch.from_numpy(kg_oneHot).float().to(device)

file_path = './data/dataset_indices.csv'
data = pd.read_csv(file_path)
x_test = data['Test_Indices'].dropna().astype(int).tolist()
print("test_size: ", len(x_test))

test_dataset = presDataset(pS_array[x_test], pH_array[x_test])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchSize, shuffle=False)

model = BSGAM(64, args.att_head_num, args.att_drop).to(device)
model.load_state_dict(torch.load('./model/best.pt'))

criterion = CustomBCEWithLogitsLoss()

epsilon = 1e-13
sh_data = sh_data.to(device)
ss_data = ss_data.to(device)
hh_data = hh_data.to(device)

model.eval()
test_loss = 0

test_p5 = 0
test_p10 = 0
test_p20 = 0

test_r5 = 0
test_r10 = 0
test_r20 = 0

for tsid, thid in test_loader:
    outputs = model(sh_tensor, s_tensor, h_tensor, sh_data.edge_index, ss_data.edge_index, hh_data.edge_index, tsid,
                    kg_oneHot, True)

    test_loss += criterion(outputs, thid, args.BCE_L).item()

    for i, hid in enumerate(thid):
        trueLabel = (hid == 1).nonzero().flatten()

        top5 = torch.topk(outputs[i], 5)[1]
        count = sum([1 for m in top5 if m in trueLabel])
        test_p5 += count / 5
        test_r5 += count / len(trueLabel)

        top10 = torch.topk(outputs[i], 10)[1]
        count = sum([1 for m in top10 if m in trueLabel])
        test_p10 += count / 10
        test_r10 += count / len(trueLabel)

        top20 = torch.topk(outputs[i], 20)[1]
        count = sum([1 for m in top20 if m in trueLabel])
        test_p20 += count / 20
        test_r20 += count / len(trueLabel)

print("----------------------------------------------------------------------------------------------------")
print('test_loss: ', test_loss / len(test_loader))
print('p5-10-20:', test_p5 / len(x_test), test_p10 / len(x_test), test_p20 / len(x_test))
print('r5-10-20:', test_r5 / len(x_test), test_r10 / len(x_test), test_r20 / len(x_test))
print('f1_5-10-20: ',
      2 * (test_p5 / len(x_test)) * (test_r5 / len(x_test)) / (
              (test_p5 / len(x_test)) + (test_r5 / len(x_test)) + epsilon),
      2 * (test_p10 / len(x_test)) * (test_r10 / len(x_test)) / (
              (test_p10 / len(x_test)) + (test_r10 / len(x_test)) + epsilon),
      2 * (test_p20 / len(x_test)) * (test_r20 / len(x_test)) / (
              (test_p20 / len(x_test)) + (test_r20 / len(x_test)) + epsilon))
