import numpy as np
from sklearn.preprocessing import RobustScaler

def robust_scale(X):
    sc = RobustScaler()
    Xs = sc.fit_transform(X)
    return Xs, sc

def normalize01(x):
    x = np.asarray(x, dtype=float)
    if np.all(~np.isfinite(x)) or np.nanstd(x) == 0:
        return np.zeros_like(x, dtype=float)
    # rank-based percentile -> [0,1]
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0, 1, len(x))
    return ranks
