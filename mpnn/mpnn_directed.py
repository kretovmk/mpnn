
import deepchem as dc
import torch.optim as optim
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchfold import Fold
from rdkit import Chem
from torch.autograd import Variable
from collections import OrderedDict

# Some parts of code taken from https://github.com/deepchem/deepchem/blob/master/contrib/mpnn/mpnn.py
# MIT License

class MPNNdirected:
    """
    Performs full message passing procedure.
    """
    def __init__(self, R, U, V, E, t):
        assert t > 0, 'ERROR: incorrect T value.'
        self.params = []
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

    def make_opt_step(self, batch_x, batch_y, t):
        self.opt.zero_grad()
        loss = Variable(torch.zeros(1, 1))
        for i in range(len(batch_x)):
            smile = batch_x[i]
            y_true = batch_y[i]
            g, h = self.get_features_from_smiles(smile)
            g2, h2 = self.get_features_from_smiles(smile)
            for k in range(0, t):
                self.single_message_pass_dyn_batched(g, h, k)
            y_pred = self.R(h, h2)
            loss += (y_pred - y_true) * (y_pred - y_true) / Variable(torch.FloatTensor([len(batch_x)])).view(1, 1)
        loss.backward()
        self.opt.step()
        return loss.data[0][0]

    def _construct_message_passing_func(self, R, U, V, E, t):
        self.R = R
        self.E = E
        for i in range(t):
            setattr(self, 'V_{}'.format(i), copy.deepcopy(V))
            setattr(self, 'U_{}'.format(i), copy.deepcopy(U))

    def _add_params(self, t):
        self.params += list(self.R.parameters())
        self.params += list(self.E.parameters())
        for i in range(t):
            self.params += list(getattr(self, 'V_{}'.format(i)).parameters())
            self.params += list(getattr(self, 'U_{}'.format(i)).parameters())

    def get_features_from_smiles(self, smile):
        g = OrderedDict({})  # edges
        h = OrderedDict({})  # atoms
        molecule = Chem.MolFromSmiles(smile)
        for i in range(0, molecule.GetNumAtoms()):
            atom_i = molecule.GetAtomWithIdx(i)
            features = dc.feat.graph_features.atom_features(atom_i).astype(np.float32)
            h[i] = Variable(torch.FloatTensor(features)).view(1, 75)
            for j in range(0, molecule.GetNumAtoms()):
                e_ij = molecule.GetBondBetweenAtoms(i, j)
                if e_ij != None:
                    e_ij = list(map(lambda x: 1 if x == True else 0, dc.feat.graph_features.bond_features(e_ij)))
                    e_ij = Variable(torch.FloatTensor(e_ij).view(1, 6))
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

    def mcat(self, x1, x2, x3):
        return torch.cat((x1, x2, x3), 1)

    def mtanh(self, x):
        return F.tanh(x)

    def single_message_pass_dyn_batched(self, g, h, k):
        fold = Fold()
        for v in g.keys():  # iterate over atoms
            neighbors = g[v]   # list of tuples of the form (e_vw, w)
            for neighbor in neighbors:
                e_vw = neighbor[0]  # bond feature (between v and w)
                w = neighbor[1]  # atom w number
                #m_w = getattr(self, 'V_{}'.format(k))(h[w])   # h[w] is feature of atom w
                m_w = fold.add('V_{}'.format(k), h[w])
                #m_e_vw = self.E(e_vw)
                m_e_vw = fold.add('E', e_vw)
                #reshaped = torch.cat((h[v], m_w, m_e_vw), 1)
                reshaped = fold.add('mcat', h[v], m_w, m_e_vw)
                #h[v] = F.tanh(getattr(self, 'U_{}'.format(k))(reshaped))
                tmp = fold.add('U_{}'.format(k), reshaped)
                h[v] = fold.add('mtanh', tmp)

        result = fold.apply(self, [list(h.values())])[0].split(1, dim=0)
        for i, r in enumerate(result):
            h[i] = r




