
import torch
import numpy as np
import deepchem as dc
import torch.utils.data as data

from utils.utils import CUDA_wrapper
from collections import OrderedDict
from torch.autograd import Variable
from deepchem.utils import ScaffoldGenerator
from rdkit import Chem



class DatasetSmiles(data.Dataset):
    """
    Class that contains data for training or test. Inherited from Pytorch abstract class.
    """
    def __init__(self, filename, cuda=False, scaffold=False, filter_dots=True, filter_atoms=True):
        self.filename = filename
        self.cuda = cuda
        self.filter_dots = filter_dots
        self.filter_atoms = filter_atoms
        self.scaffold = scaffold
        self.x = None
        self.y = None
        self._load_dataset()
        self._filter_data()
        self._calc_features()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (x, target) where target is index of the target class.
        """
        return self.filtered_x[index], self.filtered_y[index]

    def __len__(self):
        return len(self.filtered_x)

    def _load_dataset(self):
        """Load data from file."""
        f = open(self.filename, 'r')
        self.x = []
        self.y = []
        c = 0
        for line in f:
            splits = line[:-1].split(',')
            self.x.append(splits[-2])
            self.y.append(float(splits[-1]))
            c += 1
        f.close()
        print('File \"{}\" read. In total {} lines.'.format(self.filename, c))

    def _filter_data(self):
        """Filtering data from single atoms and errors (just dot instead of molecular structure)."""
        if (not self.filter_atoms) and (not self.filter_dots):
            print('Nothing filtered.')
            return
        self.filtered_x = []
        self.filtered_y = []
        # filtering and generating sca
        for smile, label in zip(self.x, self.y):
            molecule = Chem.MolFromSmiles(smile)
            if molecule.GetNumAtoms() <= 1 and self.filter_atoms:
                continue
            if '.' in smile and self.filter_dots:
                continue
            self.filtered_x.append(smile)
            self.filtered_y.append(torch.Tensor([label]))
        print('Data filtered, in total {} smiles deleted'.format(len(self.x) - len(self.filtered_x)))

    def _calc_features(self):
        self.filtered_x = [self._get_features_from_smile(x, cuda=self.cuda) for x in self.filtered_x]
        self.filtered_y = [CUDA_wrapper(y, cuda=self.cuda) for y in self.filtered_y]
        print('Features calculated and datasets prepared. Number of items in dataset: {}'.format(len(self.filtered_x)))

    def _generate_scaffold(self, smile, include_chirality=False):
        """Compute the Bemis-Murcko scaffold for data.test SMILES string."""
        mol = Chem.MolFromSmiles(smile)
        engine = ScaffoldGenerator(include_chirality=include_chirality)
        scaffold = engine.get_scaffold(mol)
        return scaffold

    def _get_features_from_smile(self, smile, cuda=False):
        """Calculate features."""
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
