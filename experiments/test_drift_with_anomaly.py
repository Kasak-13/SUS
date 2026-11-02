import sys, os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config
from src.preprocess import load_and_preprocess
from src.synthetic_drift import inject_gradual_drift
from src.sus_detector import SUSDetector

def main():
    data_real = load_and_preprocess(config.DATA_FILE)
    X_train = data_real['X_train']
    X_stream_A = data_real['X_stream'][:173]
    
    print("=== ISSUE 1: Reconciling the Anomaly Score Contradiction ===")
    rates = [0.0008, 0.0010, 0.0015]
    for rate in rates:
        print(f"\n--- Rate: {rate} ---")
        X_stream_B = inject_gradual_drift(X_stream_A, drift_rate=rate)
        detector = SUSDetector()
        detector.fit(X_train)
        
        crosses = []
        for i, x in enumerate(X_stream_B):
            score, decision = detector.update_and_predict(x)
            
            # To get the controller fraction manually:
            buffer = detector.controller.anomaly_buffer
            frac = sum(1 for s in buffer if s > detector.controller.anomaly_threshold) / len(buffer) if len(buffer) > 0 else 0
            
            if score > 0.58:
                crosses.append((i, score, frac, decision))
                
        print(f"Total points crossing 0.58 raw score: {len(crosses)}")
        for i, s, f, d in crosses[:5]: # just print first 5
            print(f"  Step {i}: Raw Score = {s:.4f} | Aggregate Fraction = {f:.2f} | Decision = {d}")
        if len(crosses) > 5:
            print("  ...")
            
    print("\n=== ISSUE 2: Catching a Real Anomaly during Drift ===")
    # Construct Scenario: Drift (0.001) + Spike at step 120-125
    X_stream_B = inject_gradual_drift(X_stream_A, drift_rate=0.0010).copy()
    
    # Inject an anomaly: multiply the values by 5 for a 5-step window
    X_stream_B[120:125] *= 5
    
    detector2 = SUSDetector()
    detector2.fit(X_train)
    
    decisions = []
    print("Decisions around the anomaly injection (Steps 115 to 140):")
    for i, x in enumerate(X_stream_B):
        score, decision = detector2.update_and_predict(x)
        decisions.append(decision)
        if 115 <= i <= 140:
            print(f"  Step {i}: Score={score:.4f}, Decision={decision}")

if __name__ == '__main__':
    main()
