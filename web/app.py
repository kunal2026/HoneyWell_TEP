# web/app.py
from __future__ import annotations
import os
import math
import tempfile
from datetime import datetime
from typing import Tuple, Dict, Any, List


from flask import Flask, render_template, request, send_file, redirect, url_for, flash
import pandas as pd
import numpy as np

# helper algorithm modules (existing in web/algorithms/)
from algorithms import (
    blend_scores,
    rank_algorithms,
    autoencoder_available,
)

# sklearn for fast-mode train-on-sample strategy and PCA scatter
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# plotting utilities (written earlier at web/algorithms/visualize.py)
from algorithms.visualize import plot_score_hist, plot_top_features_bar, plot_pca_scatter

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret")
BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

ALGORITHMS = [
    ("auto", "Auto Select (best)"),
    ("ensemble_mean", "Ensemble (mean)"),
    ("ensemble_max", "Ensemble (max)"),
    ("iso", "Isolation Forest"),
    ("mahal", "Mahalanobis (robust)"),
    ("ocsvm", "One-Class SVM"),
    ("lof", "Local Outlier Factor"),
    ("pca", "PCA Reconstruction Error"),
    ("knn", "kNN Distance"),
    ("zscore", "Robust Z-Score"),
    ("ae", "Autoencoder (if available)"),
]


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Keep numeric columns only; interpolate & fill; drop constant columns.
    Returns (clean_numeric_df, X_numpy, feature_name_list)
    """
    num = df.select_dtypes(include=[np.number]).copy()
    num.replace([np.inf, -np.inf], np.nan, inplace=True)
    num = num.interpolate(limit_direction="both").fillna(method="ffill").fillna(method="bfill")
    nunique = num.nunique()
    keep = nunique[nunique > 1].index.tolist()
    if len(keep) == 0:
        return num[keep], np.empty((0, 0)), []
    X = num[keep].values.astype(float)
    return num[keep], X, list(keep)


# --- Utility: normalize to 0..100 ---
def normalize_to_0_100(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return a
    mn = np.nanmin(a)
    mx = np.nanmax(a)
    if not np.isfinite(mn) or not np.isfinite(mx) or math.isclose(mx, mn):
        return np.zeros_like(a)
    return ((a - mn) / (mx - mn) * 100.0).astype(float)


# --- Per-model scoring & per-sample contributions (local attributions) ---
def compute_model_scores_and_contributions(
    model_key: str,
    X_full: np.ndarray,
    mode: str = "balanced",
    sample_size_threshold: int = 2000,
    ae_epochs: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a single model and return:
      - scores (n_rows,) higher -> more anomalous
      - contributions (n_rows, n_features): non-negative importance-like values per feature per row

    model_key: one of "iso","mahal","ocsvm","lof","pca","knn","zscore","ae"
    mode: "fast" | "balanced" | "accurate"
    """
    n_rows, n_features = X_full.shape
    # default fallbacks
    scores = np.zeros(n_rows, dtype=float)
    contributions = np.zeros((n_rows, n_features), dtype=float)

    # SAMPLE selection for training (fast mode): train on subset but predict full
    if mode == "fast" and n_rows > sample_size_threshold:
        idx_train = np.random.RandomState(42).choice(n_rows, size=sample_size_threshold, replace=False)
        X_train = X_full[idx_train]
    else:
        X_train = X_full

    try:
        # #########################
        # Isolation Forest
        # #########################
        if model_key == "iso":
            iso = IsolationForest(n_estimators=150 if mode != "fast" else 100,
                                  contamination="auto", random_state=42, n_jobs=-1)
            iso.fit(X_train)
            raw = -iso.score_samples(X_full)  # higher => more anomalous
            scores = raw

            # contributions:
            # - Accurate mode: try shap if available (TreeExplainer)
            if mode == "accurate":
                try:
                    import shap
                    expl = shap.TreeExplainer(iso)
                    shap_vals = expl.shap_values(X_full)
                    # shap_vals may be list for multioutput; ensure array
                    if isinstance(shap_vals, list):
                        shap_vals = np.asarray(shap_vals[0])
                    contributions = np.abs(shap_vals)
                except Exception:
                    # fallback: deviation from mean weighted by score magnitude
                    contributions = np.abs(X_full - X_train.mean(axis=0)[None, :]) * (np.abs(raw)[:, None] + 1e-8)
            else:
                # fast/balanced: simple proxy contributions = abs deviation from median
                contributions = np.abs(X_full - np.median(X_train, axis=0)[None, :])

        # #########################
        # Mahalanobis (Ledoit-Wolf)
        # #########################
        elif model_key == "mahal":
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(X_train)
            P = lw.precision_
            mu = lw.location_ if hasattr(lw, "location_") else X_train.mean(axis=0)
            d = X_full - mu[None, :]
            # squared Mahalanobis distances
            dist2 = ((d @ P) * d).sum(axis=1).clip(min=0)
            scores = dist2
            # per-feature contributions approx: squared standardized residuals
            # compute diagonal diag(P) * d**2 as proxy
            proxy = (d ** 2) * (np.abs(np.diag(P))[None, :] + 1e-12)
            contributions = proxy

        # #########################
        # One-Class SVM
        # #########################
        elif model_key == "ocsvm":
            from sklearn.svm import OneClassSVM
            oc = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
            # fit on train subset
            oc.fit(X_train)
            raw = -oc.decision_function(X_full).ravel()
            scores = raw
            # contributions: no cheap exact per-feature attribution => use abs deviation proxy
            contributions = np.abs(X_full - X_train.mean(axis=0)[None, :]) * (np.abs(raw)[:, None] + 1e-8)

        # #########################
        # LOF
        # #########################
        elif model_key == "lof":
            from sklearn.neighbors import LocalOutlierFactor
            # novelty=True allows .score_samples after fit (note: requires sklearn>=0.22)
            lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination="auto")
            lof.fit(X_train)
            raw = -lof.score_samples(X_full)
            scores = raw
            contributions = np.abs(X_full - np.median(X_train, axis=0)[None, :]) * (np.abs(raw)[:, None] + 1e-8)

        # #########################
        # PCA reconstruction error
        # #########################
        elif model_key == "pca":
            # choose number of components by mode
            n_comp = min(max(2, n_features // 2), 10) if mode == "fast" else min(max(2, n_features // 2), 20)
            pca = PCA(n_components=min(n_comp, n_features - 1), random_state=42)
            pca.fit(X_train)
            Z = pca.transform(X_full)
            Xhat = pca.inverse_transform(Z)
            per_feat_err = (X_full - Xhat) ** 2
            scores = per_feat_err.sum(axis=1)
            contributions = per_feat_err

        # #########################
        # kNN distance (k-th nearest neighbor)
        # #########################
        elif model_key == "knn":
            k = min(20, max(2, X_train.shape[0] // 50))
            nn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
            nn.fit(X_train)
            dists, idxs = nn.kneighbors(X_full, n_neighbors=k, return_distance=True)
            kth = dists[:, -1]
            scores = kth
            # contribution proxy: absolute difference from mean of neighbors per feature
            neigh_mean = np.zeros_like(X_full)
            for i in range(X_full.shape[0]):
                neigh_mean[i] = X_train[idxs[i]].mean(axis=0)
            contributions = np.abs(X_full - neigh_mean)

        # #########################
        # Robust Z-score (per-feature, aggregated)
        # #########################
        elif model_key == "zscore":
            med = np.median(X_train, axis=0)
            mad = np.median(np.abs(X_train - med[None, :]), axis=0) + 1e-9
            z = (X_full - med[None, :]) / mad[None, :]
            scores = np.sqrt((z ** 2).sum(axis=1))
            contributions = np.abs(z)  # per-feature abs z-value

        # #########################
        # Autoencoder
        # #########################
        elif model_key == "ae":
            # try to use algorithms.autoencoder.score which returns row_err, per_feature_err
            try:
                from algorithms.autoencoder import score as ae_score_fn, available as ae_available_fn
                if ae_available_fn():
                    # choose epochs by mode
                    epochs = 8 if mode == "fast" else (20 if mode == "balanced" else 50)
                    row_errs, per_feature_err = ae_score_fn(X_full, epochs=epochs)
                    row_errs = np.asarray(row_errs, dtype=float).ravel()
                    if row_errs.shape[0] != n_rows:
                        row_errs = np.zeros(n_rows)
                    scores = row_errs
                    pf = np.asarray(per_feature_err, dtype=float)
                    if pf.ndim == 2 and pf.shape[0] == n_rows and pf.shape[1] == n_features:
                        contributions = pf
                    else:
                        contributions = np.abs(X_full - X_train.mean(axis=0)[None, :])
                else:
                    # TF unavailable: fallback to zscore-like
                    med = np.median(X_train, axis=0)
                    mad = np.median(np.abs(X_train - med[None, :]), axis=0) + 1e-9
                    z = (X_full - med[None, :]) / mad[None, :]
                    scores = np.sqrt((z ** 2).sum(axis=1))
                    contributions = np.abs(z)
            except Exception:
                # fallback
                med = np.median(X_train, axis=0)
                mad = np.median(np.abs(X_train - med[None, :]), axis=0) + 1e-9
                z = (X_full - med[None, :]) / mad[None, :]
                scores = np.sqrt((z ** 2).sum(axis=1))
                contributions = np.abs(z)

        else:
            # unknown model -> zeros
            scores = np.zeros(n_rows, dtype=float)
            contributions = np.zeros((n_rows, n_features), dtype=float)

    except Exception as e:
        # If anything unexpected fails, return zeros but keep shapes
        scores = np.zeros(n_rows, dtype=float)
        contributions = np.zeros((n_rows, n_features), dtype=float)

    # ensure no NaNs
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    contributions = np.nan_to_num(contributions, nan=0.0, posinf=0.0, neginf=0.0)

    return scores, contributions


# compute all models (or only selected) with importances
def compute_all_scores_and_importances(X: np.ndarray, feature_names: List[str], mode: str = "balanced",
                                       requested_models: List[str] | None = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Run multiple models and return dictionaries:
      scores_dict[name] -> 1D array (n_rows,)
      contributions_dict[name] -> 2D array (n_rows, n_features)
    requested_models: list of keys e.g. ["iso","ae"] or None to run all.
    """
    all_keys = ["iso", "mahal", "ocsvm", "lof", "pca", "knn", "zscore", "ae"]
    keys = requested_models if requested_models is not None else all_keys

    scores_dict: Dict[str, np.ndarray] = {}
    contributions_dict: Dict[str, np.ndarray] = {}

    for k in keys:
        s, c = compute_model_scores_and_contributions(k, X, mode=mode)
        # normalize scores to 0..1 (before final 0..100 mapping later)
        s01 = (s - np.nanmin(s)) / ((np.nanmax(s) - np.nanmin(s)) or 1.0)
        scores_dict[k] = s01
        # scale contributions per-row to be relative (make them non-negative)
        # keep raw contributions matrix — we'll use magnitudes to pick top features per row
        contributions_dict[k] = np.abs(c)

    return scores_dict, contributions_dict


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        algo = request.form.get("algorithm") or "auto"   # "auto","iso","ae",...
        mode = request.form.get("mode") or "balanced"    # "fast","balanced","accurate"
        show_plots = request.form.get("show_plots", "no") == "yes"
        show_features = request.form.get("show_features", "no") == "yes"

        if not file or file.filename == "":
            flash("Please choose a CSV file.")
            return redirect(url_for("index"))

        fname = file.filename
        safe_fname = "".join(c for c in fname if c.isalnum() or c in (" ", ".", "_", "-")).strip()
        in_path = os.path.join(UPLOAD_DIR, safe_fname)
        file.save(in_path)

        try:
            raw = pd.read_csv(in_path)
            num_df, X, feature_names = preprocess(raw)
            if X.size == 0 or len(feature_names) == 0:
                raise ValueError("No usable numeric columns found after cleaning.")

            # Compute model set to run
            if algo == "auto":
                requested_models = None  # compute all, then choose best
            elif algo in ("ensemble_mean", "ensemble_max"):
                requested_models = None
            else:
                # map UI alg keys to internal model keys
                mapping = {
                    "iso": ["iso"],
                    "mahal": ["mahal"],
                    "ocsvm": ["ocsvm"],
                    "lof": ["lof"],
                    "pca": ["pca"],
                    "knn": ["knn"],
                    "zscore": ["zscore"],
                    "ae": ["ae"],
                }
                requested_models = mapping.get(algo, None)

            # Run models (fast/balanced/accurate)
            scores_dict, contributions_dict = compute_all_scores_and_importances(X, feature_names, mode=mode, requested_models=requested_models)

            # Choose final scores according to selection
            if algo == "auto":
                ranking = rank_algorithms(scores_dict)
                best_name = ranking[0][0] if len(ranking) > 0 else list(scores_dict.keys())[0]
                final_scores = scores_dict[best_name]
                final_contrib = contributions_dict.get(best_name, np.zeros_like(next(iter(contributions_dict.values()))))
                chosen_label = dict(ALGORITHMS).get(best_name, best_name)
            elif algo == "ensemble_mean":
                final_scores = blend_scores(scores_dict, mode="mean")
                # contributions aggregated by mean
                stacked = np.vstack(list(contributions_dict.values()))
                final_contrib = np.mean(stacked, axis=0) if stacked.size > 0 else np.zeros((X.shape[0], X.shape[1]))
                chosen_label = "Ensemble (mean)"
            elif algo == "ensemble_max":
                final_scores = blend_scores(scores_dict, mode="max")
                stacked = np.vstack(list(contributions_dict.values()))
                final_contrib = np.max(stacked, axis=0) if stacked.size > 0 else np.zeros((X.shape[0], X.shape[1]))
                chosen_label = "Ensemble (max)"
            else:
                # single requested model
                key = requested_models[0] if requested_models else "iso"
                final_scores = scores_dict.get(key, np.zeros(X.shape[0]))
                final_contrib = contributions_dict.get(key, np.zeros((X.shape[0], X.shape[1])))
                chosen_label = dict(ALGORITHMS).get(key, key)

            # Normalize final_scores to 0..100
            score100 = normalize_to_0_100(final_scores)

            # For per-row top-7 features use final_contrib (n_rows x n_features)
            # If contributions are (n_rows, n_features), proceed. If it's aggregated (1D), broadcast.
            if final_contrib.ndim == 1:
                final_contrib = np.tile(final_contrib[None, :], (X.shape[0], 1))
            # safeguard shapes
            if final_contrib.shape != (X.shape[0], X.shape[1]):
                final_contrib = np.zeros((X.shape[0], X.shape[1]))

            # For each row, pick indices of top-7 features by contribution value
            top_names_matrix = []
            for i in range(X.shape[0]):
                row_contrib = final_contrib[i]
                # if all zeros, fallback to abs deviation from median
                if np.allclose(row_contrib, 0):
                    row_contrib = np.abs(X[i] - np.median(X, axis=0))
                idx = np.argsort(row_contrib)[::-1][:7]
                top_names = [feature_names[j] for j in idx]
                # pad if fewer than 7 features
                while len(top_names) < 7:
                    top_names.append("")
                top_names_matrix.append(top_names)

            # Build output DataFrame aligned to original raw rows.
            out = raw.copy()
            # create full-length vector (NaN for rows that had no numeric features)
            full_scores = np.full(len(raw), np.nan, dtype=float)
            numeric_idx = num_df.index.to_numpy()
            if len(numeric_idx) == len(score100):
                full_scores[numeric_idx] = score100
            else:
                # If mismatch, attempt to fill top portion (should not happen normally)
                full_scores[:len(score100)] = score100
            out["abnormality_score"] = full_scores

            # Add top_feature_1..7 columns aligned to numeric rows (others NaN)
            for j in range(7):
                col = np.full(len(raw), "", dtype=object)
                # put names into numeric positions
                for pos_i, raw_idx in enumerate(numeric_idx[: len(top_names_matrix)]):
                    col[raw_idx] = top_names_matrix[pos_i][j]
                out[f"top_feature_{j+1}"] = col

            # Save scored CSV
            out_path = os.path.join(UPLOAD_DIR, f"scored_{safe_fname}")
            out.to_csv(out_path, index=False)

            # Generate plots if requested (plots use numeric-only rows)
            if show_plots:
                try:
                    # histogram of numeric scores (not NaN)
                    numeric_score100 = score100
                    plot_score_hist(numeric_score100, os.path.join(PLOTS_DIR, "score_hist.png"),
                                    title=f"Anomaly score distribution — {chosen_label}")
                    # aggregate importance per feature (mean of absolute contributions across rows)
                    agg_imp = final_contrib.mean(axis=0)
                    plot_top_features_bar(feature_names, agg_imp, os.path.join(PLOTS_DIR, "top_features.png"),
                                          topk=7, title=f"Top features (aggregate) — {chosen_label}")
                    # pca scatter
                    plot_pca_scatter(X, final_scores, os.path.join(PLOTS_DIR, "pca_scatter.png"),
                                     feature_names=feature_names)
                except Exception as e:
                    flash(f"Plot generation warning: {e}")

            preview = out.head(20).to_html(classes="data", index=False)
            ranking = rank_algorithms(scores_dict)

            return render_template("result.html",
                                   filename=safe_fname,
                                   algorithm_label=chosen_label,
                                   table_html=preview,
                                   ranking=ranking,
                                   download_name=f"scored_{safe_fname}",
                                   mode=mode)
        except Exception as e:
            flash(str(e))
            return redirect(url_for("index"))

    return render_template("index.html", algorithms=ALGORITHMS)


@app.route("/download/<name>")
def download(name):
    path = os.path.join(UPLOAD_DIR, name)
    if not os.path.exists(path):
        return "File not found", 404
    return send_file(path, as_attachment=True, download_name=name, mimetype="text/csv")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
