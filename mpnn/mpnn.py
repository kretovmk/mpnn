
import deepchem as dc
import torch.optim as optim
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from rdkit import Chem
from torch.autograd import Variable
from collections import OrderedDict

# Some parts of code taken from https://github.com/deepchem/deepchem/blob/master/contrib/mpnn/mpnn.py
# MIT License

class MPNN:
    """
    Performs full message passing procedure.
    """
    def __init__(self, M_start, M_hid, U_start, U_hid, R, t):
        assert t > 0, 'ERROR: incorrect T value.'
        self.params = []
        self._construct_message_passing_func(M_start, M_hid, U_start, U_hid, R, t)
        self._add_params()
        self.opt = optim.Adam(self.params, lr=1e-3)

    def make_opt_step(self, batch, t):
        x, y = batch
        self.opt.zero_grad()
        loss = Variable(torch.zeros(1, 1))
        for i in range(len(x)):
            smile = x[i]
            y_true = y[i]
            g, h = self.get_features_from_smiles(smile)
            for k in range(0, t):
                self.single_message_pass(g, h, k)
            h_concat = Variable(torch.cat([vv.data for kk, vv in h.items()], dim=0))
            y_pred = self.R(h_concat.sum(dim=0, keepdim=True))       # TODO: replace sum -- just for debug
            loss += (y_pred - y_true) ** 2 / Variable(torch.FloatTensor([len(x)])).view(1, 1)
        print(loss.data[0][0])
        loss.backward()
        self.opt.step()
        return loss.data[0][0]

    def _construct_message_passing_func(self, M_start, M_hid, U_start, U_hid, R, t):
        self.R = R
        self.U, self.M = {}, {}
        self.M[0] = M_start
        self.U[0] = U_start
        for i in range(1, t):
            self.M[i] = copy.deepcopy(M_hid)
            self.U[i] = copy.deepcopy(U_hid)

    def _add_params(self):
        self.params += list(self.R.parameters())
        for k, v in self.M.items():
            self.params += list(v.parameters())
        for k, v in self.U.items():
            self.params += list(v.parameters())

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
        h_before = {}
        for key, value in h.items():
            h_before[key] = Variable(copy.deepcopy(value.data))
        for v in g.keys():
            neighbors = g[v]
            m_v_k = None
            for neighbor in neighbors:
                e_vw = neighbor[0]  # bond feature (between v and w)
                w = neighbor[1]  # atom w feature
                if m_v_k is None:
                    m_v_k = self.M[k](torch.cat((h_before[v], h_before[w], e_vw), dim=1))
                else:
                    m_v_k += self.M[k](torch.cat((h_before[v], h_before[w], e_vw), dim=1))
            h[v] = self.U[k](torch.cat((h_before[v], m_v_k), dim=1))




