
import deepchem as dc
import torch.optim as optim
import copy
import torch
import torch.nn.functional as F
import numpy as np

from torchfold import Fold
from rdkit import Chem
from utils import CUDA_wrapper
from torch.autograd import Variable
from collections import OrderedDict

# Some parts of code taken from https://github.com/deepchem/deepchem/blob/master/contrib/mpnn/mpnn.py
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
        g, h = self.get_features_from_smiles(x)
        for k in range(0, t):
            self.single_message_pass(g, h, k)
        h_concat = Variable(torch.cat([vv.data for kk, vv in h.items()], dim=0))
        y_pred = self.R(h_concat.sum(dim=0, keepdim=True))
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

    def get_features_from_smiles(self, smile, cuda=False):
        g = OrderedDict({})  # edges
        h = OrderedDict({})  # atoms
        molecule = Chem.MolFromSmiles(smile)
        for i in range(0, molecule.GetNumAtoms()):
            atom_i = molecule.GetAtomWithIdx(i)
            features = dc.feat.graph_features.atom_features(atom_i).astype(np.float32)
            atom_var = Variable(CUDA_wrapper(torch.FloatTensor(features), cuda).view(1, 75))
            h[i] = atom_var
            for j in range(0, molecule.GetNumAtoms()):
                e_ij = molecule.GetBondBetweenAtoms(i, j)
                if e_ij != None:
                    e_ij = list(map(lambda x: 1 if x == True else 0, dc.feat.graph_features.bond_features(e_ij)))
                    edge_var = Variable(CUDA_wrapper(torch.FloatTensor(e_ij), cuda).view(1, 6))
                    e_ij = edge_var
                    if i not in g:
                        g[i] = []
                        g[i].append((e_ij, j))
        return g, h

    def single_message_pass(self, g, h, k):
        for v in g.keys():  # iterate over atoms
            neighbors = g[v]   # list of tuples of the form (e_vw, w)
            for neighbor in neighbors:
                e_vw = neighbor[0]  # bond feature (between v and w)
                w = neighbor[1]  # atom w number
                m_w = getattr(self, 'V_{}'.format(k))(h[w])   # h[w] is feature of atom w
                m_e_vw = self.E(e_vw)
                reshaped = torch.cat((h[v], m_w, m_e_vw), 1)
                h[v] = F.tanh(getattr(self, 'U_{}'.format(k))(reshaped))

    def fold_non_lin(self, x):
        return F.tanh(x)

    def single_message_pass_dyn_batched(self, g, h, k, fold):
        for v in g.keys():  # iterate over atoms
            neighbors = g[v]   # list of tuples of the form (e_vw, w)
            for neighbor in neighbors:
                e_vw = neighbor[0]  # bond feature (between v and w)
                w = neighbor[1]  # atom w number
                m_w = fold.add('V_{}'.format(k), h[w])
                m_e_vw = fold.add('E', e_vw)
                res = fold.add('U_{}'.format(k), h[v], m_w, m_e_vw)
                h[v] = fold.add('fold_non_lin', res)

    def make_opt_step_batched(self, batch_x, batch_y, t):
        self.opt.zero_grad()
        fold = Fold()
        folded_nodes = []
        for i in range(len(batch_x)):
            g, h = batch_x[i]
            for k in range(t):
                self.single_message_pass_dyn_batched(g, h, k, fold)
            folded_nodes.append(list(h.values()))
        results = fold.apply(self, folded_nodes)
        y_pred = torch.cat([self.R(x) for x  in results], dim=0)
        y_true = Variable(torch.FloatTensor(batch_y).view(-1, 1))
        loss = torch.sum((y_pred - y_true) **2 / len(batch_x), dim=0, keepdim=True)
        loss.backward()
        self.opt.step()
        return loss.data[0][0]
