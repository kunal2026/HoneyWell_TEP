
import numpy as np
from sklearn.utils import shuffle

def permutation_importance_score(model_score_fn, X, n_repeats=5, random_state=42):
    """Compute permutation importance w.r.t. anomaly score function (higher score -> more anomalous).
    model_score_fn: function(X)->scores (1d array)
    Returns mean absolute change in score per feature (higher => more important).
    """
    rng = np.random.default_rng(random_state)
    baseline = model_score_fn(X)
    base_abs = np.abs(baseline).mean()
    importances = np.zeros(X.shape[1], dtype=float)
    for j in range(X.shape[1]):
        diffs = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            s = model_score_fn(Xp)
            diffs.append(np.abs(s - baseline).mean())
        importances[j] = np.mean(diffs)
    # normalize to sum=1
    tot = importances.sum()
    if tot <= 0 or not np.isfinite(tot):
        return np.zeros_like(importances)
    return importances / tot
