# 2017 DeepCrystal Technologies - Patrick Hop
#
# Data loading a splitting file
#
# MIT License - have fun!!
# ===========================================================

import os
import random
from collections import OrderedDict

import deepchem as dc
from deepchem.utils import ScaffoldGenerator
from deepchem.utils.save import log
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

random.seed(2)
np.random.seed(2)
torch.manual_seed(2)








