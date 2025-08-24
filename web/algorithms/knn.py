import numpy as np
from sklearn.neighbors import NearestNeighbors
from .utils import normalize01

def score(X, k=20):
    k = min(k, max(2, X.shape[0]-1))
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    dists, _ = nn.kneighbors(X, n_neighbors=k)
    kth = dists[:, -1]  # distance to k-th neighbor
    return normalize01(kth)
