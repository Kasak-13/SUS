import sys, os
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config
from src.preprocess import load_and_preprocess
from src.static_detector import IsolationForestDetector
from src.synthetic_drift import inject_gradual_drift
from src.drift_monitor import DriftMonitor

def main():
    data_real = load_and_preprocess(config.DATA_FILE)
    X_train = data_real['X_train']
    X_stream_A = data_real['X_stream'][:173]
    
    detector = IsolationForestDetector()
    detector.fit(X_train)
    
    rates = [0.015, 0.010, 0.005, 0.002, 0.001]
    
    for rate in rates:
        X_stream_B = inject_gradual_drift(X_stream_A, drift_rate=rate)
        
        # Anomaly scores
        anom_scores = detector.score(X_stream_B)
        
        # Drift scores
        drift_monitor = DriftMonitor()
        drift_scores = []
        for x in X_stream_B:
            s = drift_monitor.update_and_check(x)
            if s > 0:
                drift_scores.append(s)
                
        print(f"Rate: {rate}")
        print(f"  Max Anomaly Score (needs to be < 0.58): {np.max(anom_scores):.4f}")
        print(f"  Mean KS Drift Score (needs to be > 0.75): {np.mean(drift_scores):.4f}")
        print("-" * 40)

if __name__ == '__main__':
    main()
