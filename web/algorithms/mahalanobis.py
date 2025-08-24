import numpy as np
from sklearn.covariance import LedoitWolf
from .utils import normalize01

def score(X):
    lw = LedoitWolf().fit(X)
    P = lw.precision_
    mu = X.mean(axis=0)
    d = X - mu
    dist2 = ((d @ P) * d).sum(axis=1).clip(min=0)
    return normalize01(dist2)
