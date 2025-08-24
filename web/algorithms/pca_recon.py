import numpy as np
from sklearn.decomposition import PCA
from .utils import normalize01

def score(X, n_components=10, random_state=42):
    n_components = min(n_components, max(2, min(X.shape)-1))
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X)
    Z = pca.transform(X)
    Xhat = pca.inverse_transform(Z)
    err = ((X - Xhat)**2).sum(axis=1)
    return normalize01(err)
