
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def plot_score_hist(scores, outpath, title='Anomaly score distribution'):
    ensure_dir(os.path.dirname(outpath))
    plt.figure(figsize=(6,4))
    plt.hist(scores, bins=60, edgecolor='k', alpha=0.7)
    plt.title(title)
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_top_features_bar(feature_names, importances, outpath, topk=7, title='Top features'):
    ensure_dir(os.path.dirname(outpath))
    idx = np.argsort(importances)[::-1][:topk]
    names = [feature_names[i] for i in idx]
    vals = importances[idx]
    plt.figure(figsize=(6,4))
    plt.barh(range(len(names))[::-1], vals, edgecolor='k')
    plt.yticks(range(len(names))[::-1], names)
    plt.title(title)
    plt.xlabel('Normalized importance')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_pca_scatter(X, scores, outpath, feature_names=None, topk=0):
    ensure_dir(os.path.dirname(outpath))
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)
    # color by score
    plt.figure(figsize=(6,5))
    sc = plt.scatter(Z[:,0], Z[:,1], c=scores, cmap='RdYlBu_r', s=8, alpha=0.8)
    plt.colorbar(sc, label='Anomaly score')
    plt.title('PCA 2D scatter colored by score')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
