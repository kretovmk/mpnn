
import torch.nn as nn
import torch.nn.functional as F


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
            self.out = nn.Linear(self.hid, self.out_atom)
        else:
            self.out = nn.Linear(self.inp_size, self.out_atom)

    def forward(self, x):
        if self.hid:
            out = F.relu(self.linear(x))
            return self.out(out)
        else:
            return self.out(x)