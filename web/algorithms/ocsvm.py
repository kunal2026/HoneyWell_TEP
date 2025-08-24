import numpy as np
from sklearn.svm import OneClassSVM
from .utils import normalize01

def score(X, random_state=42):
    n = X.shape[0]
    # Subsample for fit if huge (speed)
    if n > 8000:
        idx = np.random.default_rng(random_state).choice(n, 8000, replace=False)
        Xfit = X[idx]
    else:
        Xfit = X
    oc = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
    oc.fit(Xfit)
    raw = -oc.decision_function(X).ravel()  # higher = more anomalous
    return normalize01(raw)
