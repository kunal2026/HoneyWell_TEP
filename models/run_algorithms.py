import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor

def run_all_algorithms(df):
    results = {}

    # Basic preprocessing
    df = df.select_dtypes(include=['float64', 'int64']).dropna()
    X = df.values

    # Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=42)
    results['Isolation Forest'] = iso.fit_predict(X).tolist()

    # One-Class SVM
    svm = OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)
    results['One-Class SVM'] = svm.fit_predict(X).tolist()

    # Elliptic Envelope
    ee = EllipticEnvelope(contamination=0.05)
    results['Elliptic Envelope'] = ee.fit_predict(X).tolist()

    # Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    results['Local Outlier Factor'] = lof.fit_predict(X).tolist()

    return results
