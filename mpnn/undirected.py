
import torch.nn as nn
import torch.nn.functional as F



import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

VAR = 0.01

class Rd(nn.Module):
    """
    Readout function ('R' in original article).
    """
    def __init__(self, inp_size, hid_size):
        super(Rd, self).__init__()
        self.inp_size = inp_size
        self.hid_size = hid_size
        self.linear = nn.Linear(self.inp_size, self.hid_size)
        self._init_params(self.linear)
        self.out = nn.Linear(self.hid_size, 1)
        self._init_params(self.out)

    def forward(self, h):
        out = torch.sum(self.linear(h), dim=0, keepdim=True)
        return F.sigmoid(self.out(out))

    def _init_params(self, layer):
        layer.weight.data.normal_(0, VAR)
        layer.bias.data.fill_(0.)


class Md(nn.Module):
    """
    Function 'M' in original article.
    """
    def __init__(self, inp_size, out_size):
        super(Md, self).__init__()
        self.inp_size = inp_size
        self.out_size = out_size
        self.linear = nn.Linear(self.inp_size, self.out_size)
        self._init_params(self.linear)

    def forward(self, x1, x2, x3):
        x = torch.cat((x1, x2, x3), 1)
        return F.tanh(self.linear(x))


    def _init_params(self, layer):
        layer.weight.data.normal_(0, VAR)
        layer.bias.data.fill_(0.)



class Ud(nn.Module):
    """
    Function 'U' in original article.
    """
    def __init__(self, inp_size):
        super(Ud, self).__init__()
        self.inp_size = inp_size
        self.linear = nn.Linear(self.inp_size, self.inp_size)
        self._init_params(self.linear)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        return self.linear(x)


    def _init_params(self, layer):
        layer.weight.data.normal_(0, VAR)
        layer.bias.data.fill_(0.)



# class RfuncMLP(nn.Module):
#     """
#     Readout function (R in ref.article).
#     Takes as input hidden states of atom in the form (N_atoms, N_features).
#     Perform summarization over atoms' axis and then make non-linear transformation.
#     """
#     def __init__(self, inp_size, hid):
#         super(RfuncMLP, self).__init__()
#         self.inp_size = inp_size
#         self.hid = hid
#         if self.hid:
#             self.linear = nn.Linear(self.inp_size, self.hid)
#             self._init(self.linear)
#             self.out = nn.Linear(self.hid, 1)
#             self._init(self.out)
#         else:
#             self.out = nn.Linear(self.inp_size, 1)
#             self._init(self.out)
#
#
#     def forward(self, x):
#         #print(x)
#         if self.hid:
#             out = F.relu(self.linear(x))
#             return F.sigmoid(self.out(out))
#         else:
#             return F.sigmoid(self.out(x))
#
#     def _init(self, layer):
#         layer.weight.data.normal_(0, 0.1)
#         layer.bias.data.fill_(0)
#
#
# class UfuncMLP(nn.Module):
#     """
#     Vertex update function (U_t in the ref. article) in the form of MLP.
#     Takes as input concatenated vector of [his state of atom, calculated candidate M state of atom]
#     """
#     def __init__(self, inp_atom_features, inp_atom_m_state, out_size_atom, hid_size=None):
#         super(UfuncMLP, self).__init__()
#         self.inp_atom = inp_atom_features
#         self.inp_m_atom = inp_atom_m_state
#         self.out_atom = out_size_atom
#         self.hid = hid_size
#         self.inp_size = self.inp_atom + self.inp_m_atom
#         if self.hid:
#             self.linear = nn.Linear(self.inp_size, self.hid)
#             self._init(self.linear)
#             self.out = nn.Linear(self.hid, self.out_atom)
#             self._init(self.out)
#         else:
#             self.out = nn.Linear(self.inp_size, self.out_atom)
#             self._init(self.out)
#
#
#     def forward(self, x):
#         if self.hid:
#             out = F.relu(self.linear(x))
#             return self.out(out)
#         else:
#             return self.out(x)
#
#
#     def _init(self, layer):
#         layer.weight.data.normal_(0, 0.1)
#         layer.bias.data.fill_(0)
#
#
# class MfuncMLP(nn.Module):
#     """
#     Message passing function (M_t in ref. article) in the form of MLP.
#     Takes as input concatenated vector of [hid state of atom, hid state of neighbour atom, hid state of edge].
#     Return new candidate state "m" for atom.
#     """
#
#     def __init__(self, inp_atom_features, inp_edge_features, out_size_atom, hid_size=None):
#         super(MfuncMLP, self).__init__()
#         self.inp_atom = inp_atom_features
#         self.inp_edge = inp_edge_features
#         self.out_atom = out_size_atom
#         self.hid = hid_size
#         self.inp_size = self.inp_atom * 2 + self.inp_edge
#         if self.hid:
#             self.linear = nn.Linear(self.inp_size, self.hid)
#             self._init(self.linear)
#             self.out = nn.Linear(self.hid, self.out_atom)
#             self._init(self.out)
#         else:
#             self.out = nn.Linear(self.inp_size, self.out_atom)
#             self._init(self.out)
#
#
#     def forward(self, x):
#         if self.hid:
#             out = F.relu(self.linear(x))
#             return self.out(out)
#         else:
#             return self.out(x)