import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary

import manifolds
from layers.hyp_layers import get_dim_act_curv, HypDeepGraphConvolution
from layers.self_layers import merge, Self_DD, Self_AD, DeepGraphConvolution


class RevLayer(nn.Module):
    def __init__(self, manifold, curvatures, dims, acts, args):
        super(RevLayer, self).__init__()
        self.manifold = manifold
        self.curvatures = curvatures
        self.dims = dims
        self.acts = acts
        self.dropout = args.dropout
        self.bias = args.bias

        self.AD = Self_AD()
        self.stack = self.rev_stack(HypDeepGraphConvolution, self.manifold, self.dims, self.acts, self.curvatures, self.dropout, self.bias)
        self.DD = Self_DD()

    def rev_stack(self, _block, manifold, dims, acts, curvatures, dropout, bias):
        """ Create stack of irevnet blocks """
        block_list = nn.ModuleList()
        self.first = True
        for i in range(1, len(dims) - 1):
            c_in, c_out = curvatures[i], curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            block_list.append(
                HypDeepGraphConvolution(
                    manifold, in_dim, out_dim, c_in, c_out, dropout, act, bias, first = self.first
                )
            )
            self.first = False
        return block_list

    def forward(self, x):
        """ irevnet forward """
        out = self.AD.forward(x)
        for block in self.stack:
            out = block.forward(out)
        out_bij = out
        out = self.DD.forward(out_bij)
        return out, out_bij

    def inverse(self, out_bij):
        """ irevnet inverse """
        out = out_bij
        for i in range(len(self.stack)):
            out = self.stack[-1-i].inverse(out)
        x = self.AD.inverse(out)
        return x

class OriRevLayer(nn.Module):
    def __init__(self, dims, in_dim, out_dim, dropout, act, bias):
        super(OriRevLayer, self).__init__()
        self.dims = dims
        self.in_dims = in_dim
        self.out_dims = out_dim
        self.acts = act
        self.dropout = dropout
        self.bias = bias

        self.AD = Self_AD()
        self.stack = self.rev_stack(DeepGraphConvolution, self.dims, self.in_dims, self.out_dims, self.dropout, self.acts, self.bias)
        self.DD = Self_DD()

    def rev_stack(self, _block, dims, in_dim, out_dim, dropout, act, bias):
        """ Create stack of irevnet blocks """
        block_list = nn.ModuleList()
        self.first = True
        for i in range(1, len(dims) - 1):
            block_list.append(
                DeepGraphConvolution(in_dim, out_dim, dropout, act, bias, first = self.first)
            )
            self.first = False
        return block_list

    def forward(self, x):
        """ irevnet forward """
        out = self.AD.forward(x)
        for block in self.stack:
            out = block.forward(out)
        out_bij = out
        out = self.DD.forward(out_bij)
        return out, out_bij

    def inverse(self, out_bij):
        """ irevnet inverse """
        out = out_bij
        for i in range(len(self.stack)):
            out = self.stack[-1-i].inverse(out)
        x = self.AD.inverse(out)
        return x