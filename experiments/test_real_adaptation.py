import sys, os
import numpy as np
from collections import Counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config
from src.preprocess import load_and_preprocess
from src.synthetic_drift import inject_gradual_drift
from src.sus_detector import SUSDetector

def main():
    data_real = load_and_preprocess(config.DATA_FILE)
    X_train = data_real['X_train']
    X_stream_A = data_real['X_stream'][:173]
    
    rates = [0.0008, 0.0010, 0.0015]
    
    for rate in rates:
        print(f"\n=== Testing Scenario B with REAL Adaptation (rate={rate}) ===")
        X_stream_B = inject_gradual_drift(X_stream_A, drift_rate=rate)
        
        detector = SUSDetector()
        detector.fit(X_train)
        
        decisions = []
        anom_scores = []
        
        for x in X_stream_B:
            score, decision = detector.update_and_predict(x)
            anom_scores.append(score)
            decisions.append(decision)
            
        valid_decisions = [d for d in decisions if d is not None]
        counts = Counter(valid_decisions)
        total = len(valid_decisions)
        
        print("Decision Breakdown:")
        for k in ['KEEP', 'ADAPT', 'HOLD']:
            c = counts.get(k, 0)
            print(f"  {k}: {c} ({(c/total)*100:.1f}%)" if total > 0 else f"  {k}: 0 (0.0%)")
            
        print(f"Max Anomaly Score During Run: {np.max(anom_scores):.4f}")
        print(f"Final Anomaly Threshold: {detector.anomaly_threshold:.4f}")
        
        # Check if score stayed comfortably below threshold throughout (ignoring natural small spikes)
        # Because we update the threshold dynamically, we check the max score against the final threshold
        # Or better yet, we just look at the decision breakdown. If HOLD is 0% or low, the hypothesis held.

if __name__ == '__main__':
    main()
