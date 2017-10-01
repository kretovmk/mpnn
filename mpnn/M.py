
import torch.nn as nn
import torch.nn.functional as F


class MfuncMLP(nn.Module):
    """
    Message passing function (M_t in ref. article) in the form of MLP.
    Takes as input concatenated vector of [hid state of atom, hid state of neighbour atom, hid state of edge].
    Return new candidate state "m" for atom.
    """

    def __init__(self, inp_atom_features, inp_edge_features, out_size_atom, hid_size=None):
        super(MfuncMLP, self).__init__()
        self.inp_atom = inp_atom_features
        self.inp_edge = inp_edge_features
        self.out_atom = out_size_atom
        self.hid = hid_size
        self.inp_size = self.inp_atom * 2 + self.inp_edge
        if self.hid:
            self.linear = nn.Linear(self.inp_size, self.hid)
            self.out = nn.Linear(self.hid, self.out_atom)
        else:
            self.out = nn.Linear(self.inp_size, self.out_atom)

    def forward(self, x):
        if self.hid:
            out = F.relu(self.linear(x))
            return self.out(out)
        else:
            return self.out(x)