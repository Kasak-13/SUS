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
        self.retrain_count = 0
        
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
        self.retrain_count += 1
        
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
                
        # 4. Safeguard training buffer: admission gates & purity verification
        should_admit = False
        if decision == 'HOLD':
            should_admit = False
        elif decision == 'KEEP' or decision is None:
            # Part A: KEEP anomaly admission gate (admit only when pred == 0)
            should_admit = (pred == 0)
        elif decision == 'ADAPT':
            # Part B: ADAPT buffer purity gate
            # Evaluate candidate buffer using active model and current threshold
            candidate_X = np.array(self.buffer_X)
            candidate_scores = -self.model.score_samples(candidate_X)
            candidate_anom_frac = float(np.mean(candidate_scores > current_threshold))
            
            if candidate_anom_frac <= self.controller.anomaly_fraction_threshold:
                # Buffer is pure: admit observation and permit retraining
                should_admit = True
                self.buffer_X.append(x)
                if len(self.buffer_X) > self.window_size:
                    self.buffer_X.pop(0)
                self._retrain()
            else:
                # Contamination detected: abort retraining
                # Safe fallback: override decision to HOLD and engage lockdown
                decision = 'HOLD'
                self.controller.hold_countdown = self.controller.hold_duration
                self.controller.consecutive_adapt = 0
                should_admit = False
                
        if should_admit and decision != 'ADAPT':
            self.buffer_X.append(x)
            if len(self.buffer_X) > self.window_size:
                self.buffer_X.pop(0)
            
        # Store state of the observation
        self.last_score = score
        self.last_pred = pred
        self.last_threshold = current_threshold
        self.last_decision = decision
        
        if return_pred:
            return score, decision, pred
            
        return SUSResult(score, decision, pred, current_threshold)
