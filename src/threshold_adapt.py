from sklearn.ensemble import IsolationForest
import numpy as np
from src import config

class ThresholdAdaptiveDetector:
    """
    Threshold-only Adaptive Detector:
    Keeps the underlying Isolation Forest model FROZEN (weights are never updated after initial fit).
    Adapts only the decision threshold based on a rolling quantile (99th percentile) of 
    recent anomaly scores over the last `window_size` observations.
    """
    def __init__(self, window_size=config.INITIAL_TRAIN_SIZE, n_estimators=100, random_state=config.RANDOM_SEED):
        self.window_size = window_size
        self.model = IsolationForest(n_estimators=n_estimators, random_state=random_state)
        self.score_buffer = []
        self.threshold = 0
        
    def fit(self, X):
        """Initial fit on historical data and freezes the model."""
        self.model.fit(X)
        # Initialize the score buffer with the training scores
        train_scores = -self.model.score_samples(X)
        self.score_buffer = list(train_scores[-self.window_size:])
        self.threshold = np.percentile(self.score_buffer, 99)
        
    def update_and_predict(self, x):
        """
        Scores the new observation, predicts based on current threshold,
        and then updates the threshold buffer.
        """
        x_2d = x.reshape(1, -1)
        score = -self.model.score_samples(x_2d)[0]
        pred = int(score > self.threshold)
        
        # Update rolling score buffer
        self.score_buffer.append(score)
        if len(self.score_buffer) > self.window_size:
            self.score_buffer.pop(0)
            
        # Update threshold
        self.threshold = np.percentile(self.score_buffer, 99)
        
        return pred, score
