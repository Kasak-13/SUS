import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.preprocess import load_and_preprocess
from src.sus_detector import SUSDetector
from experiments.audit_benchmark_integrity import manual_metrics

data = load_and_preprocess(config.DATA_FILE)
X_train = data['X_train']
X_stream_D = data['X_stream']
y_stream_D = data['y_stream']

def run_recovery_simulation(enable_recovery=False):
    np.random.seed(config.RANDOM_SEED)
    detector = SUSDetector(random_state=config.RANDOM_SEED)
    detector.fit(X_train)
    
    preds = []
    decisions = []
    consecutive_hold_steps = 0
    recovery_events = []
    
    for i, x in enumerate(X_stream_D):
        x_2d = x.reshape(1, -1)
        score = float(-detector.model.score_samples(x_2d)[0])
        current_threshold = float(detector.anomaly_threshold)
        pred = int(score > current_threshold)
        
        drift_score = float(detector.drift_monitor.update_and_check(x))
        detector.last_drift_score = drift_score
        
        decision = None
        if len(detector.drift_monitor.stream_buffer) >= detector.drift_monitor.buffer_size:
            decision = detector.controller.decide(score, drift_score)
        else:
            detector.controller.record_observation(score)
            
        # Track HOLD duration
        if decision == 'HOLD':
            consecutive_hold_steps += 1
        else:
            consecutive_hold_steps = 0
            
        # Recovery mechanism:
        # If in HOLD for a sustained period (> 50 steps), test if the current window (50 steps) is internally stationary
        # (i.e. KS distance between first half [t-50..t-25] and second half [t-25..t] is small, e.g. < 0.25)
        if enable_recovery and consecutive_hold_steps >= 50:
            stream_buf = np.array(detector.drift_monitor.stream_buffer)
            if len(stream_buf) >= 100:
                # Compare recent window halves: [75:100] vs [50:75]
                half1 = stream_buf[50:75]
                half2 = stream_buf[75:100]
                internal_ks = max(stats.ks_2samp(half1[:, f], half2[:, f])[0] for f in range(stream_buf.shape[1]))
                
                # Also check feature variance stability
                if internal_ks < 0.35:
                    # Current window is internally stationary! The fault has subsided and settled into a stable regime.
                    # Clear the old fault scores from controller's anomaly buffer and allow recovery adaptation
                    recovery_events.append(i)
                    detector.controller.anomaly_buffer = []
                    detector.controller.hold_countdown = 0
                    detector.controller.consecutive_adapt = 0
                    consecutive_hold_steps = 0
                    decision = 'ADAPT'
                    
        # Apply buffer admission
        should_admit = False
        if decision == 'HOLD':
            should_admit = False
        elif decision == 'KEEP':
            # Part A: KEEP gate
            should_admit = (pred == 0)
        elif decision is None:
            should_admit = (pred == 0)
        elif decision == 'ADAPT':
            # Part B: ADAPT gate
            should_admit = True
            detector.buffer_X.append(x)
            if len(detector.buffer_X) > detector.window_size:
                detector.buffer_X.pop(0)
            detector._retrain()
            
        if should_admit and decision != 'ADAPT':
            detector.buffer_X.append(x)
            if len(detector.buffer_X) > detector.window_size:
                detector.buffer_X.pop(0)
                
        preds.append(pred)
        decisions.append(decision)
        
    m = manual_metrics(y_stream_D, preds)
    return m, len(recovery_events), recovery_events

print("1. Running Baseline (No Recovery)...")
m_base, rec_count_base, _ = run_recovery_simulation(enable_recovery=False)
print("2. Running With Stationarity-Based Recovery...")
m_rec, rec_count_rec, events = run_recovery_simulation(enable_recovery=True)

df_res = pd.DataFrame([
    {'Config': 'Without Recovery', **m_base, 'Recovery Events': rec_count_base},
    {'Config': 'With Recovery', **m_rec, 'Recovery Events': rec_count_rec}
])
print("\n" + df_res.to_string(index=False))
if events:
    print(f"Recovery triggered at steps: {events}")
