import sys, os
import numpy as np
import pandas as pd
from scipy import stats
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config
from src.preprocess import load_and_preprocess
from src.synthetic_drift import inject_gradual_drift

def main():
    data_real = load_and_preprocess(config.DATA_FILE)
    X_train = data_real['X_train']
    
    X_stream_A = data_real['X_stream'][:173]
    X_stream_B = inject_gradual_drift(X_stream_A, drift_rate=0.015)
    
    window_A = X_stream_A[:100]
    window_B = X_stream_B[:100]
    
    print(f"Train F5 Mean: {np.mean(X_train[:, 5]):.4f}, Std: {np.std(X_train[:, 5]):.4f}")
    print(f"Clean F5 Mean: {np.mean(window_A[:, 5]):.4f}, Std: {np.std(window_A[:, 5]):.4f}")
    print(f"Drift F5 Mean: {np.mean(window_B[:, 5]):.4f}, Std: {np.std(window_B[:, 5]):.4f}")

if __name__ == '__main__':
    main()
