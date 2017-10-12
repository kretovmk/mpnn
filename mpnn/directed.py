
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

VAR = 0.01

class Rd(nn.Module):
    """
    Readout function ('readout' in ref code).
    """
    def __init__(self, inp_size, hid_size):
        super(Rd, self).__init__()
        self.inp_size = inp_size
        self.hid_size = hid_size
        self.linear = nn.Linear(self.inp_size, self.hid_size)
        self._init_params(self.linear)
        self.out = nn.Linear(self.hid_size, 1)
        self._init_params(self.out)

    def forward(self, h, h2):
        catted_reads = map(lambda x: torch.cat([h[x[0]], h2[x[1]]], 1), zip(h2.keys(), h.keys()))
        activated_reads = map(lambda x: F.relu(self.linear(x)), catted_reads)
        readout = Variable(torch.zeros(1, self.hid_size))
        for read in activated_reads:
            readout = readout + read
        return F.sigmoid(self.out(readout))

    def _init_params(self, layer):
        layer.weight.data.normal_(0, VAR)
        layer.bias.data.fill_(0.)



class Vd(nn.Module):
    """
    Function 'V' in ref code.
    """
    def __init__(self, inp_size):
        super(Vd, self).__init__()
        self.inp_size = inp_size
        self.linear = nn.Linear(self.inp_size, self.inp_size)
        self._init_params(self.linear)

    def forward(self, x):
        return self.linear(x)


    def _init_params(self, layer):
        layer.weight.data.normal_(0, VAR)
        layer.bias.data.fill_(0.)



class Ud(nn.Module):
    """
    Function 'U' in ref code.
    """
    def __init__(self, inp_size, out_size):
        super(Ud, self).__init__()
        self.inp_size = inp_size
        self.out_size = out_size
        self.linear = nn.Linear(self.inp_size, self.out_size)
        self._init_params(self.linear)

    def forward(self, x):
        return self.linear(x)


    def _init_params(self, layer):
        layer.weight.data.normal_(0, VAR)
        layer.bias.data.fill_(0.)



class Ed(nn.Module):
    """
    Function 'E' in ref code.
    """
    def __init__(self, inp_size):
        super(Ed, self).__init__()
        self.inp_size = inp_size
        self.linear = nn.Linear(self.inp_size, self.inp_size)
        self._init_params(self.linear)

    def forward(self, x):
        return self.linear(x)


    def _init_params(self, layer):
        layer.weight.data.normal_(0, VAR)
        layer.bias.data.fill_(0.)