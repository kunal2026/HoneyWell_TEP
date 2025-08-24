import numpy as np

def blend_scores(score_dict, weights=None, mode='mean'):
    keys = list(score_dict.keys())
    M = np.vstack([score_dict[k] for k in keys])  # m x n
    if weights is None:
        w = np.ones(M.shape[0])
    else:
        w = np.array([weights.get(k, 1.0) for k in keys], dtype=float)
    w = w / (w.sum() if w.sum() != 0 else 1.0)
    if mode == 'max':
        blended = (M * w[:, None]).max(axis=0)
    else:
        blended = (M * w[:, None]).sum(axis=0)
    return blended

def separation_metric(scores):
    scores = np.asarray(scores, dtype=float)
    # Contrast between upper tail and bulk: mean(top5%) - mean(25..75%)
    n = len(scores)
    if n < 40:
        return float(scores.mean())
    q = np.quantile(scores, [0.25, 0.75, 0.95])
    bulk = scores[(scores >= q[0]) & (scores <= q[1])].mean()
    tail = scores[scores >= q[2]].mean()
    return float((tail - bulk) / (abs(bulk) + 1e-9))

def rank_algorithms(score_dict):
    items = []
    for name, s in score_dict.items():
        items.append((name, separation_metric(s)))
    items.sort(key=lambda x: x[1], reverse=True)
    return items
