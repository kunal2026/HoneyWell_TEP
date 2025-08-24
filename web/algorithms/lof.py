import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from .utils import normalize01

def score(X, random_state=42):
    n = min(35, max(10, X.shape[0]//100))
    lof = LocalOutlierFactor(n_neighbors=n, contamination='auto', novelty=False)
    raw = -lof.fit_predict(X)  # returns -1,1 labels; fallback to negative outlier factor if available
    try:
        raw = -lof.negative_outlier_factor_
    except Exception:
        pass
    return normalize01(raw)
