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

# Compare raw feature distributions:
# Pre-fault normal (0 to 172)
X_pre = X_stream_D[:173]
# Fault (173 to 573)
X_fault = X_stream_D[173:574]
# Post-fault normal (574 to 747)
X_post = X_stream_D[574:]

print(f"X_pre shape: {X_pre.shape}, mean={X_pre.mean():.4f}, std={X_pre.std():.4f}")
print(f"X_fault shape: {X_fault.shape}, mean={X_fault.mean():.4f}, std={X_fault.std():.4f}")
print(f"X_post shape: {X_post.shape}, mean={X_post.mean():.4f}, std={X_post.std():.4f}")

# Score X_post using the INITIAL model (trained on X_train only)
init_detector = SUSDetector()
init_detector.fit(X_train)
init_scores_post = -init_detector.model.score_samples(X_post)
init_fp_post = sum(1 for s in init_scores_post if s > init_detector.anomaly_threshold)
print(f"\nIf scored with the INITIAL clean model (trained only on X_train, thresh={init_detector.anomaly_threshold:.4f}):")
print(f"  Post-fault scores on X_post: mean={init_scores_post.mean():.4f}, max={init_scores_post.max():.4f}")
print(f"  Post-fault False Positives on X_post: {init_fp_post} / {len(X_post)} ({init_fp_post/len(X_post)*100:.2f}%)")

# Compare with Static Isolation Forest on X_post
from src.static_detector import IsolationForestDetector
static = IsolationForestDetector()
static.fit(X_train)
static_thresh = np.percentile(static.score(X_train), 99)
static_scores_post = static.score(X_post)
static_fp_post = sum(1 for s in static_scores_post if s > static_thresh)
print(f"\nStatic Detector on X_post (thresh={static_thresh:.4f}):")
print(f"  Post-fault False Positives: {static_fp_post} / {len(X_post)} ({static_fp_post/len(X_post)*100:.2f}%)")
