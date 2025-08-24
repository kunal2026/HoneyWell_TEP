import numpy as np
from .utils import normalize01

def score(X):
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) + 1e-9
    z = (X - med) / mad
    agg = np.sqrt((z**2).sum(axis=1))
    return normalize01(agg)
