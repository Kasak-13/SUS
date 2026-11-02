import numpy as np

def inject_gradual_drift(X, drift_rate=0.005, start_idx=0, features_to_drift=None):
    """
    Injects a gradual, linear concept drift into the features.
    This simulates a legitimate change in normal behavior over time (e.g., component wear or seasonal shift).
    
    X: numpy array (shape: n_samples, n_features)
    drift_rate: amount to add per time step
    start_idx: index at which drift begins
    features_to_drift: list of column indices to drift (if None, drifts all)
    """
    X_drifted = X.copy().astype(float)
    n_samples, n_features = X_drifted.shape
    
    if features_to_drift is None:
        features_to_drift = list(range(n_features))
        
    for i in range(start_idx, n_samples):
        drift_amount = (i - start_idx) * drift_rate
        for f in features_to_drift:
            X_drifted[i, f] += drift_amount
            
    return X_drifted
