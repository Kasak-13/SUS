import numpy as np
from sklearn.ensemble import IsolationForest
from src import config
from src.controller import SUSController
from src.drift_monitor import DriftMonitor

class SUSDetector:
    def __init__(self, window_size=config.INITIAL_TRAIN_SIZE, n_estimators=100, random_state=config.RANDOM_SEED):
        self.window_size = window_size
        self.model = IsolationForest(n_estimators=n_estimators, random_state=random_state)
        
        self.buffer_X = []
        self.controller = SUSController()
        # Initialize DriftMonitor without any specific reference data (since it's sequential)
        self.drift_monitor = DriftMonitor()
        
        # Internal state
        self.anomaly_threshold = getattr(config, 'ANOMALY_THRESHOLD', 0.58)
        self.consecutive_adapts = 0
        self.last_drift_score = 0.0
        
    def fit(self, X):
        self.buffer_X = list(X[-self.window_size:])
        self._retrain()
        
    def _retrain(self):
        X_train = np.array(self.buffer_X)
        self.model.fit(X_train)
        scores = -self.model.score_samples(X_train)
        self.anomaly_threshold = np.percentile(scores, 99)
        # Update controller's threshold to match the newly fitted model
        self.controller.anomaly_threshold = self.anomaly_threshold
        
    def update_and_predict(self, x):
        # 1. Predict with current model
        x_2d = x.reshape(1, -1)
        score = -self.model.score_samples(x_2d)[0]
        
        # 2. Get drift score
        drift_score = self.drift_monitor.update_and_check(x)
        self.last_drift_score = drift_score
        
        # 3. Ask controller (only after buffer fills)
        decision = None
        if len(self.drift_monitor.stream_buffer) >= self.drift_monitor.buffer_size:
            decision = self.controller.decide(score, drift_score)
        else:
            # Manually fill anomaly buffer during warmup
            self.controller.anomaly_buffer.append(score)
            if len(self.controller.anomaly_buffer) > self.controller.window_size:
                self.controller.anomaly_buffer.pop(0)
                
        # 4. Execute decision
        self.buffer_X.append(x)
        if len(self.buffer_X) > self.window_size:
            self.buffer_X.pop(0)
            
        if decision == 'ADAPT':
            # In a real system, we might rate-limit this, but for now we execute on ADAPT
            self._retrain()
            
        return score, decision
