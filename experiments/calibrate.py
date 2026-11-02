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
    
    # Scenarios
    X_stream_A = data_real['X_stream'][:173]
    X_stream_D = data_real['X_stream']
    X_stream_B = inject_gradual_drift(X_stream_A, drift_rate=0.005)
    
    detector = IsolationForestDetector()
    detector.fit(X_train)
    
    def analyze_stream(name, stream_X):
        drift_monitor = DriftMonitor()
        scores = []
        for x in stream_X:
            score = drift_monitor.update_and_check(x)
            if score > 0.0: # Only count once buffer is full
                scores.append(score)
                
        if scores:
            print(f"--- {name} Sequential KS Drift Score ---")
            print(f"Mean: {np.mean(scores):.4f}, Max: {np.max(scores):.4f}, Min: {np.min(scores):.4f}, 90th: {np.percentile(scores, 90):.4f}, 99th: {np.percentile(scores, 99):.4f}")
            
    analyze_stream("Scenario A (Clean)", X_stream_A)
    analyze_stream("Scenario B (Drift)", X_stream_B)
    analyze_stream("Scenario D (Anomalies)", X_stream_D)

if __name__ == '__main__':
    main()
