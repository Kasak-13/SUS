import sys, os
import numpy as np
from collections import Counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config
from src.preprocess import load_and_preprocess
from src.static_detector import IsolationForestDetector
from src.synthetic_drift import inject_gradual_drift
from src.controller import SUSController
from src.drift_monitor import DriftMonitor

def main():
    data_real = load_and_preprocess(config.DATA_FILE)
    X_train = data_real['X_train']
    X_stream_A = data_real['X_stream'][:173]
    X_stream_D = data_real['X_stream']
    
    detector = IsolationForestDetector()
    detector.fit(X_train)
    
    print("=== 1. Robustness Check: Windowed-Fraction Values ===")
    def get_fractions(X_stream):
        scores = detector.score(X_stream)
        fractions = []
        for i in range(len(scores)):
            if i >= 50:
                window = scores[i-50:i]
                frac = np.mean(window > 0.58)
                fractions.append(frac)
        return fractions

    frac_A = get_fractions(X_stream_A)
    frac_D = get_fractions(X_stream_D)
    
    print("Scenario A (Clean) Fractions:")
    print(f"  Mean: {np.mean(frac_A):.4f}, Max: {np.max(frac_A):.4f}, 90th p: {np.percentile(frac_A, 90):.4f}")
    print("Scenario D (Anomalies) Fractions:")
    print(f"  Mean: {np.mean(frac_D):.4f}, Max: {np.max(frac_D):.4f}, 90th p: {np.percentile(frac_D, 90):.4f}")
    
    
    print("\n=== 2. Robustness Check: Scenario B Alternative Drift Rates ===")
    
    def run_dry(name, X_stream):
        drift_monitor = DriftMonitor()
        controller = SUSController(anomaly_threshold=0.58, drift_threshold=0.75)
        decisions = []
        for i, x in enumerate(X_stream):
            score = detector.score(x.reshape(1, -1))[0]
            drift_score = drift_monitor.update_and_check(x)
            
            if i >= 2 * config.WINDOW_SIZE:
                decision = controller.decide(score, drift_score)
            else:
                decision = None
                controller.anomaly_buffer.append(score)
                if len(controller.anomaly_buffer) > controller.window_size:
                    controller.anomaly_buffer.pop(0)
            decisions.append(decision)
            
        valid_decisions = [d for d in decisions if d is not None]
        counts = Counter(valid_decisions)
        total = len(valid_decisions)
        
        print(f"{name}:")
        for k in ['KEEP', 'ADAPT', 'HOLD']:
            c = counts.get(k, 0)
            print(f"  {k}: {c} ({(c/total)*100:.1f}%)")
            
    X_stream_B_0008 = inject_gradual_drift(X_stream_A, drift_rate=0.0008)
    X_stream_B_0015 = inject_gradual_drift(X_stream_A, drift_rate=0.0015)
    
    run_dry("Scenario B (rate=0.0008)", X_stream_B_0008)
    print("-")
    run_dry("Scenario B (rate=0.0015)", X_stream_B_0015)

if __name__ == '__main__':
    main()
