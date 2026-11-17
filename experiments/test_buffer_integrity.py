import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.sus_detector import SUSDetector
from src.controller import SUSController

class TestBufferIntegrityAndAdmission(unittest.TestCase):
    def setUp(self):
        np.random.seed(config.RANDOM_SEED)
        self.n_features = 8
        self.X_train = np.random.normal(0, 1, (config.INITIAL_TRAIN_SIZE, self.n_features))
        
    def test_01_hold_observations_strictly_excluded_from_buffer(self):
        """Verify that any observation received when decision == 'HOLD' is NEVER admitted to buffer_X."""
        detector = SUSDetector()
        detector.fit(self.X_train)
        
        # Warm up
        for _ in range(100):
            detector.update_and_predict(np.random.normal(0, 1, self.n_features))
            
        initial_buf_len = len(detector.buffer_X)
        
        # Inject persistent extreme fault observations to trigger HOLD
        hold_points = []
        for _ in range(60):
            x_fault = np.random.normal(50.0, 5.0, self.n_features)
            res = detector.update_and_predict(x_fault)
            if res.decision == 'HOLD':
                hold_points.append(x_fault)
                
        self.assertGreaterEqual(len(hold_points), 10, "Persistent fault must trigger HOLD")
        
        # Check that none of the hold_points exist in buffer_X
        for hp in hold_points:
            for b in detector.buffer_X:
                self.assertFalse(np.array_equal(hp, b), "Observation received during HOLD was admitted to buffer_X!")

    def test_02_buffer_admission_during_clean_keep_and_adapt(self):
        """Verify that legitimate clean observations are admitted to buffer_X during KEEP and ADAPT."""
        detector = SUSDetector()
        detector.fit(self.X_train)
        
        # Feed clean baseline observations (should be KEEP once warmed up)
        for _ in range(110):
            x = np.random.normal(0, 1, self.n_features)
            detector.update_and_predict(x)
            
        self.assertEqual(len(detector.buffer_X), detector.window_size, "buffer_X must maintain full window size on clean data")
        
        # Feed gradual drift to trigger ADAPT
        adapt_detected = False
        for _ in range(70):
            x_drift = np.random.normal(1.2, 0.4, self.n_features)
            res = detector.update_and_predict(x_drift)
            if res.decision == 'ADAPT':
                adapt_detected = True
                # Newly adapted point should be admitted
                self.assertTrue(any(np.array_equal(x_drift, b) for b in detector.buffer_X), "ADAPT observation should be admitted to buffer")
                break
                
        self.assertTrue(adapt_detected, "ADAPT should trigger on gradual drift")

    def test_03_no_accidental_training_on_future_observations(self):
        """Verify strict temporal causality: detector never predicts using a model trained on that observation."""
        detector = SUSDetector()
        detector.fit(self.X_train)
        
        for _ in range(100):
            detector.update_and_predict(np.random.normal(0, 1, self.n_features))
            
        x_target = np.random.normal(2.5, 0.5, self.n_features)
        
        # Record model and prediction before update
        score_before = float(-detector.model.score_samples(x_target.reshape(1, -1))[0])
        thresh_before = detector.anomaly_threshold
        pred_before = int(score_before > thresh_before)
        
        res = detector.update_and_predict(x_target)
        
        # Returned prediction must strictly match pred_before
        self.assertEqual(res.pred, pred_before)
        self.assertAlmostEqual(res.score, score_before, places=6)
        self.assertEqual(res.threshold, thresh_before)

if __name__ == '__main__':
    unittest.main()
