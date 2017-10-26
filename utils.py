
import numpy as np

def CUDA_wrapper(tensor, cuda):
    if cuda:
        return tensor.cuda()
    else:
        return tensor

def rolling_mean(arr, n):
    """Calculate rolling mean."""
    return np.convolve(arr, np.ones((n,)) / n, mode='valid')