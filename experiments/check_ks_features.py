import numpy as np
from scipy import stats
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config
from src.preprocess import load_and_preprocess

data = load_and_preprocess(config.DATA_FILE)
X_stream_A = data['X_stream'][:173]

w1 = X_stream_A[10:60]
w2 = X_stream_A[60:110]

print("Feature-by-feature KS between [10:60] and [60:110]:")
for col in range(X_stream_A.shape[1]):
    stat, pval = stats.ks_2samp(w1[:, col], w2[:, col])
    print(f"Col {col}: KS stat = {stat:.4f}, pval = {pval:.4e}")

