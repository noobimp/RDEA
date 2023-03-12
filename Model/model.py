import sys, os

sys.path.append(os.getcwd())
from Process.process import *
import torch as th
import torch.nn.functional as F
from torch_scatter import scatter_mean

from Process.rand5fold import *
from tools.evaluate import *
from torch_geometric.nn import GINConv, global_mean_pool
from torch.nn import Sequential, Linear, ReLU
import copy
import math

import random
from torch_geometric.utils import subgraph

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

class PriorDiscriminator(th.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.l0 = th.nn.Linear(input_dim, input_dim)
        self.l1 = th.nn.Linear(input_dim, input_dim)
        self.l2 = th.nn.Linear(input_dim, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return th.sigmoid(self.l2(h))


class FF(th.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.block = th.nn.Sequential(
            th.nn.Linear(input_dim, input_dim),
            th.nn.ReLU(),
            th.nn.Linear(input_dim, input_dim),
            th.nn.ReLU(),
            th.nn.Linear(input_dim, input_dim),
            th.nn.ReLU()
        )
        self.linear_shortcut = th.nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)


class Encoder(th.nn.Module): #5000, 64, 3
    def __init__(self, num_features, dim, num_gc_layers):
        super(Encoder, self).__init__()

        self.num_gc_layers = num_gc_layers

        self.convs = th.nn.ModuleList()
        self.bns = th.nn.ModuleList()
        self.node_att_mlp = th.nn.ModuleList()
        self.edge_att_mlp = th.nn.ModuleList()
        
        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            
            conv = GINConv(nn)
            bn = th.nn.BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)
            self.with_node_attention = 1 #args.with_node_attention
            self.node_att_mlp.append(Linear(dim, 2))
            self.edge_att_mlp.append(Linear(dim*2, 2))

    def forward(self, x, edge_index, batch):

        x_one = copy.deepcopy(x)
        xsc_one = []
        xso_one = []
        for i in range(self.num_gc_layers):
            x_one = F.relu(self.convs[i](x_one, edge_index))
            if self.with_node_attention:
                node_att = 0.5 * th.ones(x_one.shape[0], 2).to(device)
            else:
                node_att = F.softmax(self.node_att_mlp[i](x_one), dim=-1)
            xc = node_att[:, 0].view(-1, 1) * x_one
            xo = node_att[:, 1].view(-1, 1) * x_one
            xsc_one.append(xc)
            xso_one.append(xo)

        xc_pool_one = [global_mean_pool(xc_one, batch) for xc_one in xsc_one] #相当于每一层的图池化
        xc_one = th.cat(xc_pool_one, 1)
        xo_pool_one = [global_mean_pool(xo_one, batch) for xo_one in xso_one] #相当于每一层的图池化
        xo_one = th.cat(xo_pool_one, 1)
        return xo_one, th.cat(xso_one, 1), xc_one, th.cat(xsc_one, 1),

    def get_embeddings(self, data):

        with th.no_grad():
            x, edge_index, batch = data.x, data.edge_index, data.batch
            go_emb, no_emb, gc_emb, nc_emb = self.forward(x, edge_index, batch)
        return no_emb


def get_positive_expectation(p_samples, measure, average=True):
    log_2 = math.log(2.)
    Ep = log_2 - F.softplus(- p_samples)
    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure, average=True):
    log_2 = math.log(2.)

    Eq = F.softplus(-q_samples) + q_samples

    if average:
        return Eq.mean()
    else:
        return Eq


def local_global_loss_(l_enc, g_enc, edge_index, batch, measure, l_enc_pos, l_enc_dropped, l_enc_dropped_two):
    '''
    Args:
        l: Local feature map.
        g: Global features.
    Returns:
        torch.Tensor: Loss.
    '''
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]

    pos_mask = th.zeros((num_nodes, num_graphs)).to(device)
    #neg_mask = th.ones((num_nodes, num_graphs)).to(device)
    neg_mask = th.rand((num_nodes, num_graphs)).to(device)
    neg_mask[neg_mask >= (1.-4./num_graphs)] = 1 
    neg_mask[neg_mask < (1.-4./num_graphs)] = 0
    
    for nodeidx, graphidx in enumerate(batch):
        pos_mask[nodeidx][graphidx] = 1.
        neg_mask[nodeidx][graphidx] = 0.
        #以防全为0
        if graphidx > 1: neg_mask[nodeidx][graphidx-1] = 1.
        else: neg_mask[nodeidx][graphidx+1] = 1.
    num_neg = th.count_nonzero(neg_mask)
    
    res = th.mm(l_enc, g_enc.t())
    res_two = th.mm(l_enc_pos, g_enc.t())
    res_three = th.mm(l_enc_dropped, g_enc.t())
    res_four = th.mm(l_enc_dropped_two, g_enc.t())

    E_pos = get_positive_expectation((res + res_two + res_three + res_four) * pos_mask / 4, measure,
                                     average=False).sum()
    E_pos = E_pos / num_nodes

    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / num_neg #(num_nodes * (num_graphs - 1))

    return E_neg - E_pos


class Net(th.nn.Module): #64, 3
    def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
        super(Net, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(5000, hidden_dim, num_gc_layers)

        self.local_d = FF(self.embedding_dim)  # Feed forward layer
        self.global_d = FF(self.embedding_dim)  # Feed forward layer

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, th.nn.Linear):
                th.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data):

        x, edge_index, dropped_edge_index, batch, num_graphs, mask = data.x, data.edge_index, data.dropped_edge_index, data.batch, max(data.batch) + 1, data.mask
        edge_sub, _ = subgraph(mask, edge_index)  # generate subgraph's edge index
        node_mask = th.ones((x.size(0), 1)).to(device)
        for i in range(x.size(0)):
            if random.random() >= 0.8:
                node_mask[i] = 0
        node_mask[data.rootindex] = 1
        x_pos_two = x * node_mask  # mask node

        go, no, gc, nc = self.encoder(x, edge_index, batch)  # y->num_graphs x dim; M->num_nodes x dim
        go1, no1,  gc1, nc1 = self.encoder(x, edge_sub, batch)  # subgraph
        go2, no2, gc2, nc2 = self.encoder(x, dropped_edge_index, batch)  # drop edge
        go3, no3, gc3, nc3 = self.encoder(x_pos_two, edge_index, batch)  # mask_node

        go_enc = self.global_d(go)  # feed forward
        #gc_enc = self.global_d(gc) #?
        
        l_enc = self.local_d(no)  # feed forward
        l_enc1 = self.local_d(no1)
        l_enc2 = self.local_d(no2)
        l_enc3 = self.local_d(no3)

        measure = 'JSD'
        local_global_loss = local_global_loss_(l_enc, go_enc, edge_index, batch, measure, l_enc1, l_enc2, l_enc3)

        return local_global_loss


class Classfier(th.nn.Module): #64*3, 64, 4
    def __init__(self, in_feats, hid_feats, num_classes):
        super(Classfier, self).__init__()
        self.linear_one = th.nn.Linear(5000 * 2, 2 * hid_feats)
        self.linear_two = th.nn.Linear(2 * hid_feats, hid_feats)
        self.linear_three = th.nn.Linear(in_feats, hid_feats)

        self.linear_transform = th.nn.Linear(hid_feats * 2, 4)
        self.prelu = th.nn.PReLU()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, th.nn.Linear):
            th.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, embed, data):
        ori = scatter_mean(data.x, data.batch, dim=0)
        root = data.x[data.rootindex]
        ori = th.cat((ori, root), dim=1)
        ori = self.linear_one(ori)
        ori = F.dropout(input=ori, p=0.5, training=self.training)
        ori = self.prelu(ori)
        ori = self.linear_two(ori)
        ori = F.dropout(input=ori, p=0.5, training=self.training)
        ori = self.prelu(ori)

        x = scatter_mean(embed, data.batch, dim=0)
        x = self.linear_three(x)
        x = F.dropout(input=x, p=0.5, training=self.training)
        x = self.prelu(x)

        out = th.cat((x, ori), dim=1)
        out = self.linear_transform(out)
        x = F.log_softmax(out, dim=1)
        return x
