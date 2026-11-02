from sklearn.ensemble import IsolationForest
import numpy as np
from src import config

class NaiveAdaptiveDetector:
    """
    Naive Adaptive Detector:
    Continuously retrains the Isolation Forest on a sliding window of the most recent 
    `window_size` observations. Retraining is triggered every `update_interval` steps.
    No check is made for whether the data contains anomalies or drift; it blindly adapts.
    """
    def __init__(self, window_size=config.INITIAL_TRAIN_SIZE, update_interval=100, n_estimators=100, random_state=config.RANDOM_SEED):
        self.window_size = window_size
        self.update_interval = update_interval
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = IsolationForest(n_estimators=n_estimators, random_state=random_state)
        
        self.buffer_X = []
        self.steps_since_update = 0
        self.threshold = 0
        
    def fit(self, X):
        """Initial fit on historical data."""
        self.buffer_X = list(X[-self.window_size:])
        self._retrain()
        
    def _retrain(self):
        """Retrains the model on the current buffer."""
        X_train = np.array(self.buffer_X)
        self.model.fit(X_train)
        scores = -self.model.score_samples(X_train)
        self.threshold = np.percentile(scores, 99)
        self.steps_since_update = 0
        
    def update_and_predict(self, x):
        """
        Updates the internal buffer with new observation x, retrains if the interval is reached,
        and returns the anomaly prediction and score.
        """
        # Predict first (using current model)
        x_2d = x.reshape(1, -1)
        score = -self.model.score_samples(x_2d)[0]
        pred = int(score > self.threshold)
        
        # Then update buffer
        self.buffer_X.append(x)
        if len(self.buffer_X) > self.window_size:
            self.buffer_X.pop(0)
            
        self.steps_since_update += 1
        if self.steps_since_update >= self.update_interval:
            self._retrain()
            
        return pred, score
