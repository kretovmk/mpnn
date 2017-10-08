
import numpy as np


def rolling_mean(arr, n):
    """Calculate rolling mean."""
    return np.convolve(arr, np.ones((n,)) / n, mode='valid')