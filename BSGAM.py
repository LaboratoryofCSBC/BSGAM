import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def softmax_1(x):
    exp_x = torch.exp(x)
    softmax_x = exp_x / (1 + exp_x.sum(dim=-1, keepdim=True))
    return softmax_x


class GCN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__(aggr='mean')
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.tanh = torch.nn.Tanh()

    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        return self.tanh(out)

    def message(self, x_j):
        x_j = self.lin(x_j)
        return x_j


class MultiHeadAtt(nn.Module):
    def __init__(self, embed_size, head_num, dropout):
        super(MultiHeadAtt, self).__init__()
        self.hid_size = embed_size
        self.head_num = head_num
        self.head_dim = self.hid_size // self.head_num

        self.W_Q = nn.Linear(self.hid_size, self.hid_size)
        self.W_K = nn.Linear(self.hid_size, self.hid_size)
        self.W_V = nn.Linear(self.hid_size, self.hid_size)

        self.fc = nn.Linear(self.hid_size, 256)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        self.scale = self.scale.to(device)

    def forward(self, query, key, value, scale=None):

        batch_size = query.shape[0]

        Q = self.W_Q(query)
        K = self.W_K(key)
        V = self.W_V(value)

        Q = Q.view(batch_size, -1, self.head_num, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.head_num, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.head_num, self.head_dim).permute(0, 2, 1, 3)

        if scale == None:
            energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        else:
            energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / scale

        attention = softmax_1(energy)

        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_size)
        x = self.fc(x)
        return x


class BSGAM(torch.nn.Module):
    def __init__(self, embedding_dim, head_num, att_drop):
        super(BSGAM, self).__init__()
        self.embedding_dim = embedding_dim

        self.SH_s_mlp = torch.nn.Linear(768, embedding_dim)
        self.SH_s_bn = torch.nn.BatchNorm1d(embedding_dim)

        self.convSH_TostudyS_1 = GCN(embedding_dim, embedding_dim)
        self.convSH_TostudyS_2 = GCN(embedding_dim, embedding_dim)

        self.SH_line_s_1 = torch.nn.Linear(embedding_dim, embedding_dim)
        self.SH_line_s_2 = torch.nn.Linear(embedding_dim, 256)
        self.SH_bn_s_1 = torch.nn.BatchNorm1d(embedding_dim)
        self.SH_bn_s_2 = torch.nn.BatchNorm1d(256)

        self.SH_h_mlp = torch.nn.Linear(768, embedding_dim)
        self.SH_h_bn = torch.nn.BatchNorm1d(embedding_dim)

        self.convSH_TostudyS_1_h = GCN(embedding_dim, embedding_dim)
        self.convSH_TostudyS_2_h = GCN(embedding_dim, embedding_dim)

        self.SH_line_h_1 = torch.nn.Linear(embedding_dim, embedding_dim)
        self.SH_line_h_2 = torch.nn.Linear(embedding_dim, 256)
        self.SH_bn_h_1 = torch.nn.BatchNorm1d(embedding_dim)
        self.SH_bn_h_2 = torch.nn.BatchNorm1d(256)

        self.SS_s_mlp = torch.nn.Linear(768, embedding_dim)
        self.SS_s_bn = torch.nn.BatchNorm1d(embedding_dim)

        self.convSS_1 = GCN(embedding_dim, embedding_dim)
        self.convSS_2 = GCN(embedding_dim, embedding_dim)

        self.SS_line_1 = torch.nn.Linear(embedding_dim, embedding_dim)
        self.SS_line_2 = torch.nn.Linear(embedding_dim, 256)
        self.SS_bn_1 = torch.nn.BatchNorm1d(embedding_dim)
        self.SS_bn_2 = torch.nn.BatchNorm1d(256)

        self.HH_h_mlp = torch.nn.Linear(768, embedding_dim)
        self.HH_h_bn = torch.nn.BatchNorm1d(embedding_dim)
        self.kg_HH_mlp = torch.nn.Linear(24, embedding_dim)
        self.kg_HH_bn = torch.nn.BatchNorm1d(embedding_dim)

        self.convHH_1 = GCN(embedding_dim, embedding_dim)
        self.convHH_2 = GCN(embedding_dim, embedding_dim)

        self.HH_line_1 = torch.nn.Linear(embedding_dim, embedding_dim)
        self.HH_line_2 = torch.nn.Linear(embedding_dim, 256)
        self.HH_bn_1 = torch.nn.BatchNorm1d(embedding_dim)
        self.HH_bn_2 = torch.nn.BatchNorm1d(256)

        self.es_bn_1 = torch.nn.BatchNorm1d(256)
        self.eh_bn_1 = torch.nn.BatchNorm1d(256)

        self.mlp = torch.nn.Linear(256, 256)
        self.SI_bn = torch.nn.BatchNorm1d(256)

        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

        self.attention_s = MultiHeadAtt(256, head_num, att_drop)
        self.attention_h = MultiHeadAtt(256, head_num, att_drop)

    def forward(self, sh_tensor, s_tensor, h_tensor, edge_index_SH, edge_index_SS, edge_index_HH, prescription,
                kgOneHot, p):

        esh0 = sh_tensor
        esh0 = self.SH_s_mlp(esh0)
        esh0 = self.SH_s_bn(esh0)
        esh0 = self.tanh(esh0)
        b0_Nsh = self.convSH_TostudyS_1(esh0, edge_index_SH)
        b1_sh = esh0 + b0_Nsh
        b1_sh = self.SH_line_s_1(b1_sh)
        b1_sh = self.SH_bn_s_1(b1_sh)
        b1_sh = self.tanh(b1_sh)

        b1_Nsh = self.convSH_TostudyS_2(b1_sh, edge_index_SH)
        b2_sh = b1_sh + b1_Nsh
        b2_sh = self.SH_line_s_2(b2_sh)
        b2_sh = self.SH_bn_s_2(b2_sh)
        b2_sh = self.tanh(b2_sh)

        esh02 = sh_tensor
        esh02 = self.SH_h_mlp(esh02)
        esh02 = self.SH_h_bn(esh02)
        esh02 = self.tanh(esh02)

        b0_Nsh2 = self.convSH_TostudyS_1_h(esh02, edge_index_SH)
        b1_sh2 = esh02 + b0_Nsh2
        b1_sh2 = self.SH_line_h_1(b1_sh2)
        b1_sh2 = self.SH_bn_h_1(b1_sh2)
        b1_sh2 = self.tanh(b1_sh2)

        b1_Nsh2 = self.convSH_TostudyS_2_h(b1_sh2, edge_index_SH)
        b2_sh2 = b1_sh2 + b1_Nsh2
        b2_sh2 = self.SH_line_h_2(b2_sh2)
        b2_sh2 = self.SH_bn_h_2(b2_sh2)
        b2_sh2 = self.tanh(b2_sh2)

        es0 = s_tensor
        es0 = self.SS_s_mlp(es0)
        es0 = self.SS_s_bn(es0)
        es0 = self.tanh(es0)
        r0_Ns = self.convSS_1(es0, edge_index_SS)
        r1_s = es0 + r0_Ns
        r1_s = self.SS_line_1(r1_s)
        r1_s = self.SS_bn_1(r1_s)
        r1_s = self.tanh(r1_s)

        r1_Ns = self.convSS_2(r1_s, edge_index_SS)
        r2_s = r1_s + r1_Ns
        r2_s = self.SS_line_2(r2_s)
        r2_s = self.SS_bn_2(r2_s)
        r2_s = self.tanh(r2_s)

        eh0 = h_tensor
        eh0 = self.HH_h_mlp(eh0)
        eh0 = self.HH_h_bn(eh0)
        eh0 = self.tanh(eh0)
        kgOneHoth0 = self.kg_HH_mlp(kgOneHot)
        kgOneHoth0 = self.kg_HH_bn(kgOneHoth0)
        kgOneHoth0 = self.tanh(kgOneHoth0)
        eh0_kg = eh0 + kgOneHoth0
        r0_Nh = self.convHH_1(eh0_kg, edge_index_HH)
        r1_h = eh0_kg + r0_Nh
        r1_h = self.HH_line_1(r1_h)
        r1_h = self.HH_bn_1(r1_h)
        r1_h = self.tanh(r1_h)

        r1_Nh = self.convHH_2(r1_h, edge_index_HH)
        r2_h = r1_h + r1_Nh
        r2_h = self.HH_line_2(r2_h)
        r2_h = self.HH_bn_2(r2_h)
        r2_h = self.tanh(r2_h)

        if p:
            query_s = (b2_sh[:390] + r2_s).view(390, 1, 256)
            key_s = torch.cat((b2_sh[:390].view(390, 1, 256), r2_s.view(390, 1, 256)), dim=1)
            value_s = torch.cat((b2_sh[:390].view(390, 1, 256), r2_s.view(390, 1, 256)), dim=1)
            es = self.attention_s(query_s, key_s, value_s).view(390, 256)
            es = self.es_bn_1(es)
            es = self.tanh(es)
            query_h = (b2_sh2[390:] + r2_h).view(811, 1, 256)
            key_h = torch.cat((b2_sh2[390:].view(811, 1, 256), r2_h.view(811, 1, 256)), dim=1)
            value_h = torch.cat((b2_sh2[390:].view(811, 1, 256), r2_h.view(811, 1, 256)), dim=1)
            eh = self.attention_h(query_h, key_h, value_h).view(811, 256)
            eh = self.eh_bn_1(eh)
            eh = self.tanh(eh)

            e_synd = torch.mm(prescription, es)
            preSum = prescription.sum(dim=1).view(-1, 1)
            e_synd_norm = e_synd / preSum
            e_synd_norm = self.mlp(e_synd_norm)
            e_synd_norm = self.SI_bn(e_synd_norm)
            e_synd_norm = self.relu(e_synd_norm)
            pre = torch.mm(e_synd_norm, eh.t())

            return pre

        else:
            return b2_sh[:390], b2_sh2[390:], r2_s, r2_h
