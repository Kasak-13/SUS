import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.preprocess import load_and_preprocess
from src.sus_detector import SUSDetector
from experiments.audit_benchmark_integrity import manual_metrics

data = load_and_preprocess(config.DATA_FILE)
X_train = data['X_train']
X_stream_D = data['X_stream']
y_stream_D = data['y_stream']

def run_experiment(use_keep_gate=False, use_purity_gate=False):
    np.random.seed(config.RANDOM_SEED)
    detector = SUSDetector(random_state=config.RANDOM_SEED)
    detector.fit(X_train)
    
    preds = []
    decisions = []
    anom_in_buf_history = []
    adapt_rejected_count = 0
    adapt_executed_count = 0
    hold_count = 0
    
    anom_indices = set(np.where(y_stream_D == 1)[0])
    
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
            
        should_admit = False
        if decision == 'HOLD':
            hold_count += 1
            should_admit = False
        elif decision == 'KEEP':
            if use_keep_gate:
                should_admit = (pred == 0)
            else:
                should_admit = True
        elif decision is None:
            if use_keep_gate:
                should_admit = (pred == 0)
            else:
                should_admit = True
        elif decision == 'ADAPT':
            if use_purity_gate:
                # Check candidate buffer purity using active model & threshold
                candidate_X = np.array(detector.buffer_X)
                candidate_scores = -detector.model.score_samples(candidate_X)
                candidate_anom_frac = float(np.mean(candidate_scores > current_threshold))
                
                if candidate_anom_frac <= detector.controller.anomaly_fraction_threshold:
                    should_admit = True
                    detector.buffer_X.append(x)
                    if len(detector.buffer_X) > detector.window_size:
                        detector.buffer_X.pop(0)
                    detector._retrain()
                    adapt_executed_count += 1
                else:
                    # Abort retraining due to contamination
                    adapt_rejected_count += 1
                    decision = 'HOLD'
                    hold_count += 1
                    detector.controller.hold_countdown = detector.controller.hold_duration
                    detector.controller.consecutive_adapt = 0
                    should_admit = False
            else:
                should_admit = True
                detector.buffer_X.append(x)
                if len(detector.buffer_X) > detector.window_size:
                    detector.buffer_X.pop(0)
                detector._retrain()
                adapt_executed_count += 1
                
        if should_admit and decision != 'ADAPT':
            detector.buffer_X.append(x)
            if len(detector.buffer_X) > detector.window_size:
                detector.buffer_X.pop(0)
                
        # Count true anomaly samples currently in buffer_X
        # Match by checking distance to X_stream_D[anom_indices]
        anoms_in_buf = 0
        if len(detector.buffer_X) > 0:
            for bx in detector.buffer_X:
                dists = np.linalg.norm(X_stream_D[list(anom_indices)] - bx, axis=1)
                if np.min(dists) < 1e-6:
                    anoms_in_buf += 1
        anom_in_buf_history.append(anoms_in_buf)
        
        preds.append(pred)
        decisions.append(decision)
        
    m = manual_metrics(y_stream_D, preds)
    return {
        'metrics': m,
        'adapt_executed': adapt_executed_count,
        'adapt_rejected': adapt_rejected_count,
        'hold_count': hold_count,
        'max_anom_in_buf': max(anom_in_buf_history),
        'mean_anom_in_buf': np.mean(anom_in_buf_history),
        'final_anom_in_buf': anom_in_buf_history[-1]
    }

print("Running Variant 1: Original Corrected SUS...")
r1 = run_experiment(use_keep_gate=False, use_purity_gate=False)

print("Running Variant 2: SUS with KEEP Admission Gate...")
r2 = run_experiment(use_keep_gate=True, use_purity_gate=False)

print("Running Variant 3: SUS with Both Safeguards (KEEP Gate + ADAPT Purity Gate)...")
r3 = run_experiment(use_keep_gate=True, use_purity_gate=True)

df_comp = pd.DataFrame([
    {'Config': '1. Base Corrected SUS', **r1['metrics'], 'ADAPT Exec': r1['adapt_executed'], 'ADAPT Rej': r1['adapt_rejected'], 'Max Anom In Buf': r1['max_anom_in_buf']},
    {'Config': '2. + KEEP Gate', **r2['metrics'], 'ADAPT Exec': r2['adapt_executed'], 'ADAPT Rej': r2['adapt_rejected'], 'Max Anom In Buf': r2['max_anom_in_buf']},
    {'Config': '3. + KEEP & ADAPT Gates', **r3['metrics'], 'ADAPT Exec': r3['adapt_executed'], 'ADAPT Rej': r3['adapt_rejected'], 'Max Anom In Buf': r3['max_anom_in_buf']},
])
print("\n" + df_comp.to_string(index=False))
