from sklearn.ensemble import IsolationForest
import numpy as np
from src import config

class IsolationForestDetector:
    def __init__(self, n_estimators=100, random_state=config.RANDOM_SEED):
        self.model = IsolationForest(n_estimators=n_estimators, random_state=random_state)
        self.is_fitted = False
        
    def fit(self, X):
        self.model.fit(X)
        self.is_fitted = True
        
    def score(self, X):
        # Higher score = more anomalous. (sklearn returns negative anomaly score for score_samples)
        return -self.model.score_samples(X)
