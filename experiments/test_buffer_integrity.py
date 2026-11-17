import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.sus_detector import SUSDetector
from src.controller import SUSController

class TestBufferIntegrityAndSafeguards(unittest.TestCase):
    def setUp(self):
        np.random.seed(config.RANDOM_SEED)
        self.n_features = 8
        self.X_train = np.random.normal(0, 1, (config.INITIAL_TRAIN_SIZE, self.n_features))

    def test_01_detected_anomalies_rejected_during_keep(self):
        """1. Verify that observations with pred == 1 during KEEP are REJECTED from buffer_X."""
        detector = SUSDetector()
        detector.fit(self.X_train)
        
        # Warmup with 100 clean observations
        for _ in range(100):
            detector.update_and_predict(np.random.normal(0, 1, self.n_features))
            
        buf_len_before = len(detector.buffer_X)
        
        # Send an isolated extreme spike (drift will be low, so raw_decision is KEEP - Quadrant 2)
        x_spike = np.random.normal(10.0, 1.0, self.n_features)
        res = detector.update_and_predict(x_spike)
        
        # Should be KEEP (isolated spike) and pred == 1 (anomalous)
        self.assertEqual(res.decision, 'KEEP')
        self.assertEqual(res.pred, 1)
        
        # Verify that x_spike was NOT admitted to buffer_X
        for b in detector.buffer_X:
            self.assertFalse(np.array_equal(x_spike, b), "Anomalous observation during KEEP must be rejected from buffer_X!")

    def test_02_clean_observations_admitted_during_keep(self):
        """2. Verify that observations with pred == 0 during KEEP are ADMITTED into buffer_X."""
        detector = SUSDetector()
        detector.fit(self.X_train)
        
        for _ in range(100):
            detector.update_and_predict(np.random.normal(0, 1, self.n_features))
            
        x_clean = np.random.normal(0.01, 0.9, self.n_features)
        res = detector.update_and_predict(x_clean)
        
        self.assertEqual(res.decision, 'KEEP')
        self.assertEqual(res.pred, 0)
        
        # Verify that x_clean IS in buffer_X (at the newest position -1)
        self.assertTrue(np.array_equal(detector.buffer_X[-1], x_clean), "Clean observation during KEEP must be admitted to buffer_X!")

    def test_03_contaminated_candidate_buffer_rejected_before_retraining(self):
        """3. Verify that if candidate buffer exceeds ANOMALY_FRACTION_THRESHOLD, retraining is ABORTED."""
        detector = SUSDetector(window_size=100)
        detector.fit(self.X_train)
        
        # Prime the stream buffer and controller anomaly buffer with 100 clean observations
        for _ in range(100):
            detector.update_and_predict(np.random.normal(0, 1, self.n_features))
            
        # Artificially inject 20% contaminated points into the middle/end of buffer_X so they are not immediately evicted
        n_contam = int(0.20 * detector.window_size)
        for i in range(n_contam):
            detector.buffer_X[-(i+1)] = np.random.normal(20.0, 1.0, self.n_features)
            
        initial_retrain_count = detector.retrain_count
        
        # Feed gradual drift to trigger ADAPT proposal (Quadrant 3)
        adapt_rejected = False
        for _ in range(60):
            x_drift = np.random.normal(1.2, 0.4, self.n_features)
            res = detector.update_and_predict(x_drift)
            if res.decision == 'HOLD' and detector.controller.consecutive_adapt == 0:
                # Contaminated candidate buffer aborted retraining!
                adapt_rejected = True
                self.assertEqual(detector.controller.hold_countdown, detector.controller.hold_duration)
                break
                
        self.assertTrue(adapt_rejected, "Contaminated candidate buffer must trigger adaptation rejection and HOLD")
        self.assertEqual(detector.retrain_count, initial_retrain_count, "Retraining must not execute on contaminated buffer")

    def test_04_clean_candidate_buffer_permitted_to_retrain(self):
        """4. Verify that a pure candidate buffer is permitted to retrain on legitimate ADAPT."""
        detector = SUSDetector()
        detector.fit(self.X_train)
        
        for _ in range(100):
            detector.update_and_predict(np.random.normal(0, 1, self.n_features))
            
        initial_retrain_count = detector.retrain_count
        
        # Feed gradual drift to legitimately trigger ADAPT with clean buffer
        adapted = False
        for _ in range(70):
            x_drift = np.random.normal(1.2, 0.4, self.n_features)
            res = detector.update_and_predict(x_drift)
            if res.decision == 'ADAPT':
                adapted = True
                self.assertGreater(detector.retrain_count, initial_retrain_count, "Retrain count should increment on clean ADAPT")
                break
                
        self.assertTrue(adapted, "Clean gradual drift must trigger ADAPT")

    def test_05_no_retraining_with_hold_observations(self):
        """5. Verify that observations received during HOLD never enter buffer_X or retraining."""
        detector = SUSDetector()
        detector.fit(self.X_train)
        
        for _ in range(100):
            detector.update_and_predict(np.random.normal(0, 1, self.n_features))
            
        hold_points = []
        for _ in range(60):
            x_fault = np.random.normal(50.0, 5.0, self.n_features)
            res = detector.update_and_predict(x_fault)
            if res.decision == 'HOLD':
                hold_points.append(x_fault)
                
        self.assertGreaterEqual(len(hold_points), 10)
        for hp in hold_points:
            for b in detector.buffer_X:
                self.assertFalse(np.array_equal(hp, b), "HOLD observation must never be present in buffer_X!")

    def test_06_no_future_observation_leakage(self):
        """6. Verify that prediction on x_t is evaluated strictly prior to buffer admission or retraining."""
        detector = SUSDetector()
        detector.fit(self.X_train)
        
        for _ in range(100):
            detector.update_and_predict(np.random.normal(0, 1, self.n_features))
            
        x_target = np.random.normal(2.5, 0.5, self.n_features)
        
        # State before
        score_before = float(-detector.model.score_samples(x_target.reshape(1, -1))[0])
        thresh_before = detector.anomaly_threshold
        pred_before = int(score_before > thresh_before)
        
        res = detector.update_and_predict(x_target)
        
        self.assertEqual(res.pred, pred_before)
        self.assertAlmostEqual(res.score, score_before, places=6)
        self.assertEqual(res.threshold, thresh_before)

    def test_07_correct_behaviour_when_adaptation_rejected(self):
        """7. Verify controller state when adaptation is rejected due to buffer contamination."""
        detector = SUSDetector(window_size=50)
        detector.fit(self.X_train)
        
        # Prime buffers
        for _ in range(100):
            detector.update_and_predict(np.random.normal(0, 1, self.n_features))
            
        # Contaminate candidate buffer
        for i in range(10):
            detector.buffer_X[i] = np.random.normal(15.0, 1.0, self.n_features)
            
        initial_retrain_count = detector.retrain_count
        
        # Drive gradual drift to trigger ADAPT
        adapt_rejected = False
        for _ in range(60):
            x_drift = np.random.normal(1.2, 0.4, self.n_features)
            res = detector.update_and_predict(x_drift)
            if res.decision == 'HOLD' and detector.controller.consecutive_adapt == 0:
                adapt_rejected = True
                self.assertEqual(detector.controller.hold_countdown, config.HOLD_DURATION)
                self.assertEqual(detector.controller.consecutive_adapt, 0)
                # Verify that x_drift was not added to buffer
                self.assertFalse(np.array_equal(detector.buffer_X[-1], x_drift))
                break
                
        self.assertTrue(adapt_rejected, "Contaminated buffer should trigger adaptation rejection")
        self.assertEqual(detector.retrain_count, initial_retrain_count)

    def test_08_post_fault_no_unsafe_learning_of_active_faults(self):
        """8. Verify that active fault state is never learned as normal baseline."""
        detector = SUSDetector()
        detector.fit(self.X_train)
        
        for _ in range(100):
            detector.update_and_predict(np.random.normal(0, 1, self.n_features))
            
        # Feed 150 persistent fault observations
        # Should stay in HOLD and NEVER adapt to the fault
        adapt_during_fault = 0
        for _ in range(150):
            x_fault = np.random.normal(30.0, 5.0, self.n_features)
            res = detector.update_and_predict(x_fault)
            if res.decision == 'ADAPT':
                adapt_during_fault += 1
                
        self.assertEqual(adapt_during_fault, 0, "Active severe fault must NEVER trigger ADAPT!")

if __name__ == '__main__':
    unittest.main()
