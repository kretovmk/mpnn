
import torch.nn as nn
import torch.nn.functional as F


class FuncR(nn.Module):
    """
    Readout function (R in ref.article).
    Takes as input hidden states of atom in the form (N_atoms, N_features).
    Perform summarization over atoms' axis and then make non-linear transformation.
    """
    def __init__(self, inp_size, hid):
        super(FuncR, self).__init__()
        self.inp_size = inp_size
        self.hid = hid
        if self.hid:
            self.linear = nn.Linear(self.inp_size, self.hid)
            self.out = nn.Linear(self.hid, 1)
        else:
            self.out = nn.Linear(self.inp_size, 1)


    def forward(self, x):
        if self.hid:
            out = F.relu(self.linear(x))
            return F.sigmoid(self.out(out))
        else:
            return F.sigmoid(self.out(x))