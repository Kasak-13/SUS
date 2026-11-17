import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.preprocess import load_and_preprocess
from src.sus_detector import SUSDetector

data = load_and_preprocess(config.DATA_FILE)
X_train = data['X_train']
X_stream_D = data['X_stream']
y_stream_D = data['y_stream']

detector = SUSDetector(random_state=config.RANDOM_SEED)
detector.fit(X_train)

for i, x in enumerate(X_stream_D):
    thresh_before = detector.anomaly_threshold
    score, decision = detector.update_and_predict(x)
    if decision == 'ADAPT':
        buf = np.array(detector.buffer_X)
        scores = -detector.model.score_samples(buf)
        frac = np.mean(scores > thresh_before)
        print(f"Step {i:3d} (y={y_stream_D[i]}): ADAPT proposed. buffer_len={len(buf)}, fraction exceeding thresh={frac:.4f} (threshold={detector.controller.anomaly_fraction_threshold})")
