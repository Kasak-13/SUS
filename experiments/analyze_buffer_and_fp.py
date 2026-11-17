import os
import sys
import numpy as np
import pandas as pd
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.preprocess import load_and_preprocess
from src.sus_detector import SUSDetector

data = load_and_preprocess(config.DATA_FILE)
X_train = data['X_train']
X_stream_D = data['X_stream']
y_stream_D = data['y_stream']
t_stream_D = data['t_stream']

print(f"Total Stream D Steps: {len(X_stream_D)}")
anom_indices = np.where(y_stream_D == 1)[0]
normal_indices = np.where(y_stream_D == 0)[0]
print(f"Anomaly indices range: min={anom_indices.min()}, max={anom_indices.max()}, count={len(anom_indices)}")
print(f"Normal indices count: {len(normal_indices)}")

# Check if anomaly is contiguous or split
diffs = np.diff(anom_indices)
split_points = np.where(diffs > 1)[0]
if len(split_points) == 0:
    print(f"Anomalies form a single contiguous block: [{anom_indices[0]}, {anom_indices[-1]}]")
else:
    print(f"Anomalies are split across multiple blocks: {split_points}")

# Trace SUS run step-by-step
sus = SUSDetector()
sus.fit(X_train)

trace = []
for i in range(len(X_stream_D)):
    x = X_stream_D[i]
    y_true = int(y_stream_D[i])
    t_val = t_stream_D[i]
    
    # Pre-step state
    thresh_before = sus.anomaly_threshold
    buf_len_before = len(sus.buffer_X)
    
    # Count true anomalies inside buffer_X before this step
    # We can check by matching against X_stream_D[anom_indices]
    anom_in_buf = 0
    if len(sus.buffer_X) > 0:
        buf_arr = np.array(sus.buffer_X)
        # Check how many points in buf_arr belong to the true anomaly set
        # Since points are continuous floats, we can compare with X_stream_D[anom_indices]
        for bx in buf_arr:
            # check if bx is in X_stream_D[anom_indices]
            # using norm distance
            dists = np.linalg.norm(X_stream_D[anom_indices] - bx, axis=1)
            if np.min(dists) < 1e-6:
                anom_in_buf += 1
                
    score, decision = sus.update_and_predict(x)
    pred = int(sus.last_pred)
    drift_score = float(sus.last_drift_score)
    admitted = (decision != 'HOLD')
    
    # Classify classification outcome
    is_tp = (y_true == 1 and pred == 1)
    is_tn = (y_true == 0 and pred == 0)
    is_fp = (y_true == 0 and pred == 1)
    is_fn = (y_true == 1 and pred == 0)
    
    trace.append({
        'step': i,
        'timestamp': str(t_val),
        'y_true': y_true,
        'pred': pred,
        'score': score,
        'thresh': thresh_before,
        'drift_score': drift_score,
        'decision': decision,
        'admitted_to_buffer': admitted,
        'anom_in_buf': anom_in_buf,
        'buf_len': len(sus.buffer_X),
        'is_tp': is_tp,
        'is_tn': is_tn,
        'is_fp': is_fp,
        'is_fn': is_fn
    })

df_trace = pd.DataFrame(trace)

# Analyze False Positives
fps = df_trace[df_trace['is_fp']]
print(f"\nTotal False Positives: {len(fps)} / {len(normal_indices)} (FPR = {len(fps)/len(normal_indices):.4f})")

# Breakdown of FPs by time phases:
# Phase 1: Pre-fault normal period (steps 0 to anom_indices.min() - 1)
pre_fault_fp = fps[fps['step'] < anom_indices.min()]
# Phase 2: During fault (None because y_true == 1 during fault)
# Phase 3: Post-fault normal period (steps > anom_indices.max())
post_fault_fp = fps[fps['step'] > anom_indices.max()]

print(f"Pre-fault FPs (Steps 0 to {anom_indices.min()-1}): {len(pre_fault_fp)} FPs out of {anom_indices.min()} normal steps (FPR = {len(pre_fault_fp)/anom_indices.min():.4f})")
post_len = len(X_stream_D) - 1 - anom_indices.max()
print(f"Post-fault FPs (Steps {anom_indices.max()+1} to {len(X_stream_D)-1}): {len(post_fault_fp)} FPs out of {post_len} normal steps (FPR = {len(post_fault_fp)/post_len:.4f})")

print("\n--- Pre-fault FP sample ---")
print(pre_fault_fp[['step', 'score', 'thresh', 'drift_score', 'decision']].head(10))

print("\n--- Post-fault FP sample ---")
print(post_fault_fp[['step', 'score', 'thresh', 'drift_score', 'decision']].head(15))

# Analyze Retraining Events
print("\n--- Retraining (ADAPT) Analysis ---")
adapt_steps = df_trace[df_trace['decision'] == 'ADAPT']
print(f"Total ADAPT executions: {len(adapt_steps)}")
for idx, row in adapt_steps.iterrows():
    print(f"Step {row['step']:3d} | y_true={row['y_true']} | Score={row['score']:.4f} | Thresh={row['thresh']:.4f} | Drift={row['drift_score']:.4f} | Anom points in buffer={row['anom_in_buf']}/{row['buf_len']} ({row['anom_in_buf']/row['buf_len']*100:.1f}%)")
