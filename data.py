
import torch
import numpy as np

from rdkit import Chem
from torch.autograd import Variable
from sklearn import preprocessing
from deepchem.utils import ScaffoldGenerator
from deepchem.utils.save import log
from rdkit import Chem



class DataPreprocessor:


    def __init__(self,
                 filename,
                 filter_dots=False,
                 filter_atoms=True,
                 ):
        self.filename = filename
        self.filter_dots = filter_dots
        self.filter_atoms = filter_atoms


    def get_data(self,
                 frac_train=.80,
                 frac_valid=.10,
                 frac_test=.10,
                 log_every_n=1000):
        self.load_dataset()
        self.filter_data()
        train_ind, valid_ind, test_ind = self.split(self.filtered_smiles, frac_train, frac_valid, frac_test, log_every_n)
        train_features = [self.filtered_smiles[i] for i in train_ind]
        train_labels = [self.filtered_labels[i] for i in train_ind]
        valid_features = [self.filtered_smiles[i] for i in valid_ind]
        valid_labels = [self.filtered_labels[i] for i in valid_ind]
        test_features = [self.filtered_smiles[i] for i in test_ind]
        test_labels = [self.filtered_labels[i] for i in test_ind]
        return train_features, train_labels, valid_features, valid_labels, test_features, test_labels


    def load_dataset(self):
        f = open(self.filename, 'r')
        self.all_smiles = []
        self.all_labels = []
        c = 0
        for line in f:
            splits = line[:-1].split(',')
            self.all_smiles.append(splits[-2])
            self.all_labels.append(float(splits[-1]))
            c += 1
        f.close()
        print('File {} read. In total {} lines.'.format(self.filename, c))


    def _generate_scaffold(self, smile, include_chirality=False):
        """Compute the Bemis-Murcko scaffold for data.test SMILES string."""
        mol = Chem.MolFromSmiles(smile)
        engine = ScaffoldGenerator(include_chirality=include_chirality)
        scaffold = engine.get_scaffold(mol)
        return scaffold


    def split(self,
              dataset,
              frac_train=.80,
              frac_valid=.10,
              frac_test=.10,
              log_every_n=1000):
        """
        Splits internal compounds into train/validation/test by scaffold.
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
        scaffolds = {}
        log("About to generate scaffolds", True)
        data_len = len(dataset)

        for ind, smiles in enumerate(dataset):

            if ind % log_every_n == 0:
                log("Generating scaffold %d/%d" % (ind, data_len), True)
            scaffold = self._generate_scaffold(smiles)
            if scaffold not in scaffolds:
                scaffolds[scaffold] = [ind]
            else:
                scaffolds[scaffold].append(ind)

        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
        scaffold_sets = [
            scaffold_set
            for (scaffold, scaffold_set) in sorted(
                scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
            ]
        train_cutoff = frac_train * len(dataset)
        valid_cutoff = (frac_train + frac_valid) * len(dataset)
        train_inds, valid_inds, test_inds = [], [], []
        log("About to sort in scaffold sets", True)
        for scaffold_set in scaffold_sets:
            if len(train_inds) + len(scaffold_set) > train_cutoff:
                if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                    test_inds += scaffold_set
                else:
                    valid_inds += scaffold_set
            else:
                train_inds += scaffold_set
        return train_inds, valid_inds, test_inds


    def filter_data(self):
        if (not self.filter_atoms) and (not self.filter_dots):
            print('Nothing filtered.')
            return
        self.filtered_smiles = []
        self.filtered_labels = []
        # filtering and generating sca
        for smile, label in zip(self.all_smiles, self.all_labels):
            molecule = Chem.MolFromSmiles(smile)
            if molecule.GetNumAtoms() <= 1 and self.filter_atoms:
                continue
            if '.' in smile and self.filter_dots:
                continue
            self.filtered_smiles.append(smile)
            self.filtered_labels.append(label)
        print('Data filtered, in total {} smiles deleted'.format(len(self.all_smiles) - len(self.filtered_smiles)))