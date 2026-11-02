import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config
from src.preprocess import load_and_preprocess
from src.static_detector import IsolationForestDetector
from src.drift_monitor import DriftMonitor
from src.controller import SUSController

data_real = load_and_preprocess(config.DATA_FILE)
X_train = data_real['X_train']
X_stream_A = data_real['X_stream'][:173]

detector = IsolationForestDetector()
detector.fit(X_train)

drift_monitor = DriftMonitor()
controller = SUSController(anomaly_threshold=0.58, drift_threshold=0.75)

decisions = []
for i, x in enumerate(X_stream_A):
    score = detector.score(x.reshape(1, -1))[0]
    drift_score = drift_monitor.update_and_check(x)
    if i >= 2 * config.WINDOW_SIZE:
        d = controller.decide(score, drift_score)
    else:
        d = None
    decisions.append((i, d, drift_score, score))

valid = [x for x in decisions if x[1] is not None]
print(f"Total valid decisions: {len(valid)}")
for i, d, ds, s in valid[:20]:
    print(f"Step {i}: decision={d}, drift={ds:.4f}, score={s:.4f}")

