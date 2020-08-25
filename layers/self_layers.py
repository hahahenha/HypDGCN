"""Euclidean layers."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


def get_dim_act(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers - 1))
    if args.task in ['lp', 'rec']:
        dims += [args.dim]
        acts += [act]
    return dims, acts


class GraphConvolution(Module):
    """
    Simple GCN layer.
    """

    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(GraphConvolution, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, input):
        x, adj = input
        hidden = self.linear.forward(x)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        if adj.is_sparse:
            support = torch.spmm(adj, hidden)
        else:
            support = torch.mm(adj, hidden)
        output = self.act(support), adj
        return output

    def extra_repr(self):
        return 'input_dim={}, output_dim={}'.format(
                self.in_features, self.out_features
        )


class Linear(Module):
    """
    Simple Linear layer with dropout.
    """

    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act

    def forward(self, x):
        hidden = self.linear.forward(x)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        out = self.act(hidden)
        return out


class FermiDiracDecoder(Module):
    """Fermi Dirac to compute edge probabilities based on distances."""

    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t

    def forward(self, dist):
        probs = 1. / (torch.exp((dist - self.r) / self.t) + 1.0)
        return probs

def split(x):
    if x.shape[0] < x.shape[1]:
        x1 = x[:, :(x.shape[1] - x.shape[0])].contiguous()
        x2 = x[:, (x.shape[1] - x.shape[0]):].contiguous()
        x3 = x1[:(x1.shape[1]), :].contiguous()
        out = x3, x2
    else:
        x1 = x[:(x.shape[1]), :].contiguous()
        x2 = x[(x.shape[1]):, :].contiguous()
        x3 = x2[:, :(x2.shape[0])].contiguous()
        out = x1, x3
    return out

def merge(x1, x2):
    dnum1 = x1.shape[0]
    dnum2 = x2.shape[0]
    if dnum1 > dnum2:
        b = torch.zeros((dnum2 - dnum1, dnum1))
        x = torch.cat((x1, b), 0)
        out = torch.cat((x, x2), 1)
    else:
        b = torch.zeros((dnum2, dnum1 - dnum2))
        x = torch.cat((x2, b), 1)
        out = torch.cat((x1, x), 0)
    return out


class ListModule(object):
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))

class Self_AD(Module):
    def __init__(self):
        super(Self_AD, self).__init__()

    def forward(self, x):
        h, adj = x
        return h, h, adj

    def inverse(self, output):
        a, b, adj = output
        return a, adj

class Self_DD(Module):
    def __init__(self):
        super(Self_DD, self).__init__()

    def forward(self, x):
        a, b, adj = x
        return b

class DeepGraphConvolution(nn.Module):
    """
    Deep graph convolution layer.
    """

    def __init__(self, in_features, out_features, dropout, act, use_bias, first=False):
        super(DeepGraphConvolution, self).__init__()
        self.dropout = dropout
        self.act = act
        self.in_features = out_features
        self.out_features = out_features

        if not first:
            self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(out_features, out_features, use_bias)

    def forward(self, input):
        x1, x2, adj = input
        hidden = self.linear.forward(x2)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        if adj.is_sparse:
            support = torch.spmm(adj, hidden)
        else:
            support = torch.mm(adj, hidden)
        Fx2 = self.act(support)
        y1 = Fx2 + x1
        y2 = x2
        return y2, y1, adj

    def inverse(self, output):
        """ bijective or injecitve block inverse """
        x2, y1, adj = output
        hidden = self.linear.forward(x2)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        if adj.is_sparse:
            support = torch.spmm(adj, hidden)
        else:
            support = torch.mm(adj, hidden)
        Fx2 = self.act(support)
        x1 = Fx2 + y1
        return x1, x2, adj