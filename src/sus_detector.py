import numpy as np
from sklearn.ensemble import IsolationForest
from src import config
from src.controller import SUSController
from src.drift_monitor import DriftMonitor

class SUSResult(tuple):
    """
    Tuple subclass representing (score, decision) for backwards compatibility,
    while providing explicit attributes for model versioning and prediction.
    """
    def __new__(cls, score, decision, pred=0, threshold=0.0):
        return super().__new__(cls, (score, decision))

    def __init__(self, score, decision, pred=0, threshold=0.0):
        self.score = score
        self.decision = decision
        self.pred = pred
        self.threshold = threshold

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
        self.last_score = 0.0
        self.last_pred = 0
        self.last_threshold = self.anomaly_threshold
        self.last_decision = None
        
    def fit(self, X):
        self.buffer_X = list(X[-self.window_size:])
        self._retrain()
        
    def _retrain(self):
        X_train = np.array(self.buffer_X)
        self.model.fit(X_train)
        scores = -self.model.score_samples(X_train)
        self.anomaly_threshold = float(np.percentile(scores, 99))
        # Update controller's threshold to match the newly fitted model
        self.controller.anomaly_threshold = self.anomaly_threshold
        
    def update_and_predict(self, x, return_pred=False):
        # 1. Predict with current model (strictly prior to any adaptation)
        x_2d = x.reshape(1, -1)
        score = float(-self.model.score_samples(x_2d)[0])
        current_threshold = float(self.anomaly_threshold)
        pred = int(score > current_threshold)
        
        # 2. Get drift score
        drift_score = float(self.drift_monitor.update_and_check(x))
        self.last_drift_score = drift_score
        
        # 3. Ask controller (only after drift buffer fills)
        decision = None
        if len(self.drift_monitor.stream_buffer) >= self.drift_monitor.buffer_size:
            decision = self.controller.decide(score, drift_score)
        else:
            # Maintain controller's rolling anomaly buffer during warmup via encapsulated API
            self.controller.record_observation(score)
                
        # 4. Safeguard training buffer: exclude contaminated observations during HOLD
        if decision != 'HOLD':
            self.buffer_X.append(x)
            if len(self.buffer_X) > self.window_size:
                self.buffer_X.pop(0)
            
        # 5. Execute decision
        if decision == 'ADAPT':
            # Execute retraining upon approved ADAPT decision
            self._retrain()
            
        # Store state of the observation
        self.last_score = score
        self.last_pred = pred
        self.last_threshold = current_threshold
        self.last_decision = decision
        
        if return_pred:
            return score, decision, pred
            
        return SUSResult(score, decision, pred, current_threshold)
