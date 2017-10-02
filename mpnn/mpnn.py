
import deepchem as dc
from rdkit import Chem

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable
from collections import OrderedDict



class MPNN:
    """
    Performs full message passing procedure.
    """

    def __init__(self, ):
        self.params = None

    def add_params(self, params):
        self.params.append(params)

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


    def message_pass(self, g, h, k):
        for v in g.keys():
            neighbors = g[v]
            for neighbor in neighbors:
                e_vw = neighbor[0]  # bond feature (between v and w)
                w = neighbor[1]  # atom w feature
                m_w = V[k](h[w])
                m_e_vw = E(e_vw)
                reshaped = torch.cat((h[v], m_w, m_e_vw), 1)
                h[v] = F.selu(U[k](reshaped))




