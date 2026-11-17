import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.preprocess import load_and_preprocess
from src.sus_detector import SUSDetector

data = load_and_preprocess(config.DATA_FILE)
X_train = data['X_train']
X_stream_D = data['X_stream']
y_stream_D = data['y_stream']

sus = SUSDetector()
sus.fit(X_train)

trace = []
for i in range(len(X_stream_D)):
    x = X_stream_D[i]
    y_true = int(y_stream_D[i])
    thresh_before = sus.anomaly_threshold
    
    score, decision = sus.update_and_predict(x)
    pred = int(sus.last_pred)
    drift = float(sus.last_drift_score)
    hold_cd = sus.controller.hold_countdown
    consec_adapt = sus.controller.consecutive_adapt
    
    # Calculate fraction of buffer that is anomalous
    anom_buffer = sus.controller.anomaly_buffer
    anom_frac = sum(1 for s in anom_buffer if s > sus.controller.anomaly_threshold) / len(anom_buffer) if len(anom_buffer) > 0 else 0.0
    
    trace.append({
        'step': i,
        'y_true': y_true,
        'pred': pred,
        'score': score,
        'thresh': thresh_before,
        'drift': drift,
        'anom_frac': anom_frac,
        'decision': decision,
        'hold_cd': hold_cd,
        'consec_adapt': consec_adapt,
        'buf_len': len(sus.buffer_X)
    })

df = pd.DataFrame(trace)

print("=== PART A: Fault Onset (Steps 170 to 235) ===")
print(df.loc[170:235, ['step', 'y_true', 'pred', 'score', 'thresh', 'drift', 'anom_frac', 'decision', 'hold_cd']].to_string(index=False))

print("\n=== PART B: Anomaly Midpoint & ADAPT Event (Steps 250 to 265) ===")
print(df.loc[250:265, ['step', 'y_true', 'pred', 'score', 'thresh', 'drift', 'anom_frac', 'decision', 'hold_cd', 'consec_adapt']].to_string(index=False))

print("\n=== PART C: Fault Termination & Recovery (Steps 570 to 620) ===")
print(df.loc[570:620, ['step', 'y_true', 'pred', 'score', 'thresh', 'drift', 'anom_frac', 'decision', 'hold_cd']].to_string(index=False))
