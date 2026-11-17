import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.sus_detector import SUSDetector

# Create dummy training data (400 samples, 8 features)
X_train = np.ones((400, 8)) * 1.0
X_stream = np.array([np.ones(8) * (i + 10) for i in range(150)])

detector = SUSDetector()
detector.fit(X_train)

print("Tracing update_and_predict across warmup (steps 95 to 105):")
for i, x in enumerate(X_stream):
    buf_len_before = len(detector.controller.anomaly_buffer)
    stream_buf_before = len(detector.drift_monitor.stream_buffer)
    
    score, decision = detector.update_and_predict(x)
    
    buf_len_after = len(detector.controller.anomaly_buffer)
    stream_buf_after = len(detector.drift_monitor.stream_buffer)
    
    if 95 <= i <= 105:
        print(f"Step {i:3d}: decision={decision}, stream_buf: {stream_buf_before}->{stream_buf_after}, anomaly_buf: {buf_len_before}->{buf_len_after}, last_score={score:.4f}")
