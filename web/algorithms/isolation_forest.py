import numpy as np
from sklearn.ensemble import IsolationForest
from .utils import normalize01

def score(X, random_state=42):
    model = IsolationForest(
        n_estimators=300, contamination='auto', random_state=random_state, n_jobs=-1
    )
    model.fit(X)
    raw = -model.score_samples(X)   # higher = more anomalous
    return normalize01(raw)
