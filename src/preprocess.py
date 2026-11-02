import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src import config

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath, sep=';', index_col='datetime', parse_dates=True)
    
    # Check monotonicity
    assert df.index.is_monotonic_increasing, "Timestamps are not strictly ordered"
    
    # ffill/bfill
    df = df.ffill().bfill()
    
    if 'anomaly' not in df.columns:
        df['anomaly'] = 0
    if 'changepoint' not in df.columns:
        df['changepoint'] = 0
        
    features = [col for col in df.columns if col not in ['anomaly', 'changepoint']]
    X = df[features].values
    y = df['anomaly'].values
    t = df.index.values
    
    X_train = X[:config.INITIAL_TRAIN_SIZE]
    X_stream = X[config.INITIAL_TRAIN_SIZE:]
    y_stream = y[config.INITIAL_TRAIN_SIZE:]
    t_stream = t[config.INITIAL_TRAIN_SIZE:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_stream_scaled = scaler.transform(X_stream)
    
    return {
        'X_train': X_train_scaled,
        'X_stream': X_stream_scaled,
        'y_stream': y_stream,
        't_stream': t_stream,
        'feature_names': features
    }
