
import copy
import torch
import torch.optim as optim

from torch.autograd import Variable
from utils.torchfold import Fold


# Some parts of code taken from https://github.com/deepchem/deepchem/blob/master/contrib/mpnn/mpnn.py
# It is heavily rewrited and added batching
# MIT License

class MPNNdirected:
    """
    Performs full message passing procedure.
    """
    def __init__(self, R, U, V, E, t, cuda=False):
        assert t > 0, 'ERROR: incorrect T value.'
        self.params = []
        self.cuda = cuda
        self._construct_message_passing_func(R, U, V, E, t)
        self._add_params(t)
        self.opt = optim.Adam(self.params, lr=1e-3)

    def forward_pass(self, x, t):
        fold, folded_nodes = self.batch_operations([x], t)
        result = fold.apply(self, folded_nodes)
        y_pred = torch.cat([self.R(x) for x in result], dim=0)
        return y_pred

    def _construct_message_passing_func(self, R, U, V, E, t):
        self.R = R
        self.E = E
        if self.cuda:
            self.R.cuda()
            self.E.cuda()
        for i in range(t):
            if self.cuda:
                setattr(self, 'V_{}'.format(i), copy.deepcopy(V).cuda())
                setattr(self, 'U_{}'.format(i), copy.deepcopy(U).cuda())
            else:
                setattr(self, 'V_{}'.format(i), copy.deepcopy(V))
                setattr(self, 'U_{}'.format(i), copy.deepcopy(U))

    def _add_params(self, t):
        self.params += list(self.R.parameters())
        self.params += list(self.E.parameters())
        for i in range(t):
            self.params += list(getattr(self, 'V_{}'.format(i)).parameters())
            self.params += list(getattr(self, 'U_{}'.format(i)).parameters())

    def single_message_pass_dyn_batched(self, g, h, k, fold):
        for v in g.keys():  # iterate over atoms
            neighbors = g[v]   # list of tuples of the form (e_vw, w)
            for neighbor in neighbors:
                e_vw = neighbor[0]  # bond feature (between v and w)
                w = neighbor[1]  # atom w number
                m_w = fold.add('V_{}'.format(k), h[w])
                m_e_vw = fold.add('E', e_vw)
                h[v] = fold.add('U_{}'.format(k), h[v], m_w, m_e_vw)

    def batch_operations(self, x, t):
        fold = Fold()
        folded_nodes = []
        for i in range(len(x)):
            g, h = x[i]
            for k in range(t):
                self.single_message_pass_dyn_batched(g, h, k, fold)
            folded_nodes.append(list(h.values()))
        return fold, folded_nodes

    def make_opt_step_batched(self, results, y_true):
        self.opt.zero_grad()
        y_pred = torch.cat([self.R(x) for x  in results], dim=0)
        y_true = Variable(torch.cat(y_true).view(-1, 1))
        if self.cuda:
            y_true = y_true.cuda()
        loss = torch.sum((y_pred - y_true) **2 / len(y_true), dim=0, keepdim=True)
        loss.backward()
        self.opt.step()
        return loss.data[0][0]
