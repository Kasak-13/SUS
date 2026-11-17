import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.preprocess import load_and_preprocess

# Let's inspect multiple SKAB files
skab_files = [
    ("Scenario D (valve1/0.csv)", "data/raw/skab_repo/data/valve1/0.csv"),
    ("valve1/1.csv", "data/raw/skab_repo/data/valve1/1.csv"),
    ("valve1/2.csv", "data/raw/skab_repo/data/valve1/2.csv"),
    ("valve2/0.csv", "data/raw/skab_repo/data/valve2/0.csv"),
]

for label, fpath in skab_files:
    if not os.path.exists(fpath):
        continue
    data = load_and_preprocess(fpath)
    X_stream = data['X_stream']
    y_stream = data['y_stream']
    
    anom_idx = np.where(y_stream == 1)[0]
    if len(anom_idx) == 0:
        continue
    start_anom = anom_idx[0]
    end_anom = anom_idx[-1]
    
    # Pre-fault
    X_pre = X_stream[:start_anom]
    # Fault
    X_fault = X_stream[start_anom:end_anom+1]
    # Post-fault
    X_post = X_stream[end_anom+1:]
    
    print(f"\n================ {label} ================")
    print(f"Total stream: {len(X_stream)} | Anomaly window: [{start_anom}, {end_anom}] (len={len(X_fault)})")
    print(f"Pre-fault len: {len(X_pre)} | Post-fault len: {len(X_post)}")
    
    # Compute feature-wise variance
    var_pre = np.var(X_pre, axis=0).mean()
    var_fault = np.var(X_fault, axis=0).mean()
    var_post = np.var(X_post, axis=0).mean() if len(X_post) > 0 else 0
    print(f"Mean feature variance: Pre={var_pre:.4f} | Fault={var_fault:.4f} | Post={var_post:.4f}")
    
    # KS test between pre and post
    if len(X_post) >= 50 and len(X_pre) >= 50:
        ks_pre_post = max(stats.ks_2samp(X_pre[:, f], X_post[:, f])[0] for f in range(X_stream.shape[1]))
        ks_fault_post = max(stats.ks_2samp(X_fault[:, f], X_post[:, f])[0] for f in range(X_stream.shape[1]))
        print(f"Max KS distance (Pre vs Post): {ks_pre_post:.4f}")
        print(f"Max KS distance (Fault vs Post): {ks_fault_post:.4f}")
