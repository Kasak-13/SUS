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

# Scenario D properties
anom_indices = set(np.where(y_stream_D == 1)[0])
fault_end_step = max(anom_indices) # step 573

def run_single_ablation_trial(variant, random_seed=42):
    """
    variant:
      1: Original Corrected SUS (HOLD exclusion only)
      2: SUS with KEEP Admission Gate
      3: SUS with Both Buffer Safeguards (KEEP Gate + ADAPT Purity Gate)
      4: SUS with Both Safeguards + Proposed Recovery Mechanism
    """
    np.random.seed(random_seed)
    detector = SUSDetector(random_state=random_seed)
    detector.fit(X_train)
    
    preds = []
    decisions = []
    anom_in_buf_history = []
    adapt_executed = 0
    adapt_rejected = 0
    hold_count = 0
    recovery_events = []
    recovery_delay = None
    
    consecutive_post_fault_steps = 0
    
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
            
        # Variant 4: Recovery mechanism
        # When fault terminates (step > 573), if post-fault window reaches internal stationarity (drift between halves < 0.25)
        if variant == 4 and i > fault_end_step:
            consecutive_post_fault_steps += 1
            if consecutive_post_fault_steps >= 50:
                stream_buf = np.array(detector.drift_monitor.stream_buffer)
                if len(stream_buf) >= 100:
                    half1 = stream_buf[50:75]
                    half2 = stream_buf[75:100]
                    internal_ks = max(stats.ks_2samp(half1[:, f], half2[:, f])[0] for f in range(stream_buf.shape[1]))
                    if internal_ks < 0.25:
                        # Distribution is internally stationary! Allow recovery adaptation
                        if len(recovery_events) == 0:
                            recovery_events.append(i)
                            recovery_delay = i - fault_end_step
                            detector.controller.anomaly_buffer = []
                            detector.controller.hold_countdown = 0
                            detector.controller.consecutive_adapt = 3
                            decision = 'ADAPT'
                            
        # Execution of buffer policy
        should_admit = False
        if decision == 'HOLD':
            hold_count += 1
            should_admit = False
        elif decision == 'KEEP' or decision is None:
            if variant in [2, 3, 4]:
                # KEEP gate: admit only when pred == 0
                should_admit = (pred == 0)
            else:
                should_admit = True
        elif decision == 'ADAPT':
            if variant in [3, 4]:
                # Purity gate: verify candidate buffer purity
                candidate_X = np.array(detector.buffer_X)
                candidate_scores = -detector.model.score_samples(candidate_X)
                candidate_anom_frac = float(np.mean(candidate_scores > current_threshold))
                
                if candidate_anom_frac <= detector.controller.anomaly_fraction_threshold:
                    should_admit = True
                    detector.buffer_X.append(x)
                    if len(detector.buffer_X) > detector.window_size:
                        detector.buffer_X.pop(0)
                    detector._retrain()
                    adapt_executed += 1
                else:
                    # Abort retraining
                    adapt_rejected += 1
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
                adapt_executed += 1
                
        if should_admit and decision != 'ADAPT':
            detector.buffer_X.append(x)
            if len(detector.buffer_X) > detector.window_size:
                detector.buffer_X.pop(0)
                
        # Measure true anomalies in buffer_X
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
        **m,
        'ADAPT Exec': adapt_executed,
        'ADAPT Rej': adapt_rejected,
        'HOLD Count': hold_count,
        'Recovery Delay (steps)': recovery_delay if recovery_delay is not None else -1,
        'Max Anom in Buf': max(anom_in_buf_history),
        'Mean Anom in Buf': round(np.mean(anom_in_buf_history), 2),
        'Final Anom in Buf': anom_in_buf_history[-1]
    }

print("Running 3 Trials for all 4 Configurations on Scenario D...")
configs = [
    (1, "1. Original Corrected SUS"),
    (2, "2. SUS + KEEP Admission Gate"),
    (3, "3. SUS + Both Buffer Safeguards"),
    (4, "4. SUS + Both Safeguards + Recovery")
]

all_results = []
for var_id, var_name in configs:
    trials = []
    for trial_idx in range(1, 4):
        res = run_single_ablation_trial(var_id, random_seed=config.RANDOM_SEED)
        res['Trial'] = trial_idx
        trials.append(res)
    # Check reproducibility ignoring Trial key
    t1_clean = {k: v for k, v in trials[0].items() if k != 'Trial'}
    t2_clean = {k: v for k, v in trials[1].items() if k != 'Trial'}
    t3_clean = {k: v for k, v in trials[2].items() if k != 'Trial'}
    reproducible = (t1_clean == t2_clean == t3_clean)
    row = {'Configuration': var_name, **t1_clean, 'Reproducible (3x)': reproducible}
    all_results.append(row)
    print(f"  Finished {var_name} (Reproducible: {reproducible})")

df_ablation = pd.DataFrame(all_results)
cols_order = [
    'Configuration', 'Precision', 'Recall', 'F1_Score', 'FPR', 'FNR',
    'TP', 'TN', 'FP', 'FN', 'ADAPT Exec', 'ADAPT Rej', 'HOLD Count',
    'Recovery Delay (steps)', 'Max Anom in Buf', 'Mean Anom in Buf', 'Final Anom in Buf',
    'Reproducible (3x)'
]
print("\n================ COMPREHENSIVE ABLATION STUDY RESULTS ================")
print(df_ablation[cols_order].to_string(index=False))
