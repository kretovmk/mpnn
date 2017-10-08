
import torch.nn as nn
import torch.nn.functional as F


class RfuncMLP(nn.Module):
    """
    Readout function (R in ref.article).
    Takes as input hidden states of atom in the form (N_atoms, N_features).
    Perform summarization over atoms' axis and then make non-linear transformation.
    """
    def __init__(self, inp_size, hid):
        super(RfuncMLP, self).__init__()
        self.inp_size = inp_size
        self.hid = hid
        if self.hid:
            self.linear = nn.Linear(self.inp_size, self.hid)
            self._init(self.linear)
            self.out = nn.Linear(self.hid, 1)
            self._init(self.out)
        else:
            self.out = nn.Linear(self.inp_size, 1)
            self._init(self.out)


    def forward(self, x):
        #print(x)
        if self.hid:
            out = F.relu(self.linear(x))
            return F.sigmoid(self.out(out))
        else:
            return F.sigmoid(self.out(x))

    def _init(self, layer):
        layer.weight.data.normal_(0, 0.1)
        layer.bias.data.fill_(0)


class UfuncMLP(nn.Module):
    """
    Vertex update function (U_t in the ref. article) in the form of MLP.
    Takes as input concatenated vector of [his state of atom, calculated candidate M state of atom]
    """
    def __init__(self, inp_atom_features, inp_atom_m_state, out_size_atom, hid_size=None):
        super(UfuncMLP, self).__init__()
        self.inp_atom = inp_atom_features
        self.inp_m_atom = inp_atom_m_state
        self.out_atom = out_size_atom
        self.hid = hid_size
        self.inp_size = self.inp_atom + self.inp_m_atom
        if self.hid:
            self.linear = nn.Linear(self.inp_size, self.hid)
            self._init(self.linear)
            self.out = nn.Linear(self.hid, self.out_atom)
            self._init(self.out)
        else:
            self.out = nn.Linear(self.inp_size, self.out_atom)
            self._init(self.out)


    def forward(self, x):
        if self.hid:
            out = F.relu(self.linear(x))
            return self.out(out)
        else:
            return self.out(x)


    def _init(self, layer):
        layer.weight.data.normal_(0, 0.1)
        layer.bias.data.fill_(0)


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
            self._init(self.linear)
            self.out = nn.Linear(self.hid, self.out_atom)
            self._init(self.out)
        else:
            self.out = nn.Linear(self.inp_size, self.out_atom)
            self._init(self.out)


    def forward(self, x):
        if self.hid:
            out = F.relu(self.linear(x))
            return self.out(out)
        else:
            return self.out(x)