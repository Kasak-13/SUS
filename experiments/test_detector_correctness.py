import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.sus_detector import SUSDetector, SUSResult
from src.controller import SUSController
from src.drift_monitor import DriftMonitor

class TestSUSDetectorCorrectness(unittest.TestCase):
    def setUp(self):
        np.random.seed(config.RANDOM_SEED)
        self.n_features = 8
        self.n_samples = config.INITIAL_TRAIN_SIZE
        # Normal baseline training data ~ N(0, 1)
        self.X_train = np.random.normal(0, 1, (self.n_samples, self.n_features))
        
    def test_01_anomaly_score_and_threshold_version_consistency(self):
        """Issue 1 & 2: Score and threshold must belong to same model version, and retraining
        must not change the threshold used for the current observation."""
        detector = SUSDetector()
        detector.fit(self.X_train)
        
        initial_threshold = detector.anomaly_threshold
        
        # Feed clean baseline until drift buffer is primed
        for i in range(100):
            x_clean = np.random.normal(0, 1, self.n_features)
            res = detector.update_and_predict(x_clean)
            self.assertEqual(detector.last_threshold, initial_threshold)
            self.assertEqual(res.threshold, initial_threshold)
            
        # Induce persistent statistical shift without high anomaly (Quadrant 3: ADAPT)
        # Shift mean moderately so drift triggers without creating massive anomalies
        # Window size is 50, so need >= 40 points for KS test to exceed 0.75
        adapted = False
        for step in range(60):
            x_shifted = np.random.normal(1.2, 0.4, self.n_features)
            thresh_before = detector.anomaly_threshold
            res = detector.update_and_predict(x_shifted)
            
            # Prediction and last_threshold must strictly use thresh_before
            self.assertEqual(detector.last_threshold, thresh_before)
            self.assertEqual(res.threshold, thresh_before)
            self.assertEqual(detector.last_pred, int(res.score > thresh_before))
            
            if res.decision == 'ADAPT':
                adapted = True
                # Newly trained model threshold may differ
                # But observation prediction must have matched thresh_before
                self.assertEqual(res.pred, int(res.score > thresh_before))
                break
                
        self.assertTrue(adapted, "ADAPT decision should trigger under sustained distribution shift")

    def test_02_prediction_coupled_to_scoring_model_version(self):
        """Issue 3: Ensure current prediction is evaluated using the model that scored it."""
        detector = SUSDetector()
        detector.fit(self.X_train)
        
        x = np.random.normal(0, 1, self.n_features)
        
        # Test backward-compatible unpack (score, decision)
        score, decision = detector.update_and_predict(x)
        self.assertIsInstance(score, float)
        
        # Test tuple attributes
        res = detector.update_and_predict(x)
        self.assertTrue(hasattr(res, 'pred'))
        self.assertTrue(hasattr(res, 'threshold'))
        self.assertTrue(hasattr(res, 'score'))
        self.assertTrue(hasattr(res, 'decision'))
        
        # Test return_pred=True
        score3, decision3, pred3 = detector.update_and_predict(x, return_pred=True)
        self.assertEqual(pred3, detector.last_pred)
        self.assertEqual(pred3, int(score3 > detector.last_threshold))

    def test_03_no_duplicate_observation_in_warmup_transition(self):
        """Issue 4: Investigate duplicate observation insertion during initial warm-up transition."""
        detector = SUSDetector()
        detector.fit(self.X_train)
        
        # Buffer sizes: drift_monitor requires 2*window_size = 100, controller window_size = 50
        history_anom_lens = []
        history_stream_lens = []
        
        # Feed 110 observations across the 100-step warmup transition boundary
        for i in range(110):
            x = np.random.normal(0, 1, self.n_features)
            score, decision = detector.update_and_predict(x)
            history_anom_lens.append(len(detector.controller.anomaly_buffer))
            history_stream_lens.append(len(detector.drift_monitor.stream_buffer))
            
            # Controller buffer must never exceed window_size (50)
            self.assertLessEqual(len(detector.controller.anomaly_buffer), detector.controller.window_size)
            # Drift stream buffer must never exceed buffer_size (100)
            self.assertLessEqual(len(detector.drift_monitor.stream_buffer), detector.drift_monitor.buffer_size)
            
            # Prior to step 100 (index 99), decision must be None (warmup)
            if i < 99:
                self.assertIsNone(decision)
            elif i == 99:
                # Transition step exactly at index 99 (100th observation)
                self.assertIsNotNone(decision)
                self.assertEqual(len(detector.drift_monitor.stream_buffer), 100)
                self.assertEqual(len(detector.controller.anomaly_buffer), 50)
                
        # Check smooth incremental progression with no duplicate jumps
        for i in range(1, 50):
            self.assertEqual(history_anom_lens[i] - history_anom_lens[i-1], 1, "Buffer must grow exactly by 1 each step")
        for i in range(50, 110):
            self.assertEqual(history_anom_lens[i], 50, "Buffer size must stay at window_size=50")

    def test_04_sliding_window_behavior_and_fifo_ordering(self):
        """Issue 5: Check buffer initialization, sliding window behaviour and observation ordering."""
        detector = SUSDetector(window_size=50)
        # Seed with known sequenced values
        X_seq = np.array([[float(i)] * self.n_features for i in range(50)])
        detector.fit(X_seq)
        
        # Check buffer_X holds exact initial order
        self.assertEqual(len(detector.buffer_X), 50)
        self.assertEqual(detector.buffer_X[0][0], 0.0)
        self.assertEqual(detector.buffer_X[-1][0], 49.0)
        
        # Push 10 new distinct vectors
        for i in range(50, 60):
            x_new = np.array([float(i)] * self.n_features)
            detector.update_and_predict(x_new)
            
        # Buffer should have evicted the oldest 10, newest should be 59.0
        self.assertEqual(len(detector.buffer_X), 50)
        self.assertEqual(detector.buffer_X[0][0], 10.0)
        self.assertEqual(detector.buffer_X[-1][0], 59.0)

    def test_05_hold_period_observations_excluded_from_retraining_buffer(self):
        """Issue 6: Model retraining must NOT include contaminated or HOLD-period observations."""
        detector = SUSDetector(window_size=100)
        detector.fit(self.X_train)
        
        # Warm up drift monitor with 100 clean observations
        for _ in range(100):
            detector.update_and_predict(np.random.normal(0, 1, self.n_features))
            
        # Inject persistent extreme fault observations (Quadrant 4: high drift + high anomaly)
        # Sustained fault for 60 steps so KS statistic on window 50 exceeds DRIFT_THRESHOLD (0.75)
        hold_count = 0
        hold_points = []
        for _ in range(60):
            x_fault = np.random.normal(100.0, 10.0, self.n_features)
            res = detector.update_and_predict(x_fault)
            if res.decision == 'HOLD':
                hold_count += 1
                hold_points.append(x_fault)
                
        self.assertGreaterEqual(hold_count, 10, "Persistent fault should trigger HOLD for multiple timesteps")
        
        # Verify that none of the observations received during HOLD were added to buffer_X
        for x_h in hold_points:
            for b in detector.buffer_X:
                self.assertFalse(np.array_equal(x_h, b), "HOLD observation was improperly stored in retraining buffer_X!")

    def test_06_data_leakage_between_training_and_evaluation(self):
        """Issue 7: Observation x_t must never be evaluated against a model trained on x_t."""
        detector = SUSDetector(window_size=50)
        detector.fit(self.X_train)
        
        # Warm up
        for _ in range(100):
            detector.update_and_predict(np.random.normal(0, 1, self.n_features))
            
        # Generate target observation
        x_target = np.random.normal(2.0, 0.2, self.n_features)
        
        # Model state prior to update
        model_before = detector.model
        score_expected = float(-model_before.score_samples(x_target.reshape(1, -1))[0])
        thresh_expected = detector.anomaly_threshold
        pred_expected = int(score_expected > thresh_expected)
        
        # Predict on x_target
        res = detector.update_and_predict(x_target)
        
        # Prediction and score must match model_before, never post-update model
        self.assertAlmostEqual(res.score, score_expected, places=6)
        self.assertEqual(res.pred, pred_expected)
        self.assertEqual(res.threshold, thresh_expected)

    def test_07_adaptation_safeguards_preserved_across_updates(self):
        """Issue 8: Model updates must strictly preserve adaptation safeguards."""
        detector = SUSDetector()
        detector.fit(self.X_train)
        
        # Verify safeguard parameters match config
        self.assertEqual(detector.controller.min_consecutive, config.MIN_CONSECUTIVE_OBSERVATIONS)
        self.assertEqual(detector.controller.hold_duration, config.HOLD_DURATION)
        
        # Force a retrain
        detector._retrain()
        
        # Safeguards must remain uncorrupted
        self.assertEqual(detector.controller.min_consecutive, 3)
        self.assertEqual(detector.controller.hold_duration, 10)
        self.assertGreater(detector.anomaly_threshold, 0.0)
        self.assertEqual(detector.controller.anomaly_threshold, detector.anomaly_threshold)

    def test_08_hold_countdown_lockdown_enforced(self):
        """Issue 8 (cont.): When HOLD triggers, hold_countdown locks down adaptation for HOLD_DURATION."""
        controller = SUSController()
        # Mock high drift and high anomaly
        controller.anomaly_buffer = [0.99] * 50
        controller.anomaly_threshold = 0.58
        
        # Initial trigger
        decision = controller.decide(0.99, 0.95)
        self.assertEqual(decision, 'HOLD')
        self.assertEqual(controller.hold_countdown, config.HOLD_DURATION)
        
        # Even if drift and anomaly abruptly drop to normal, controller must remain in HOLD for full countdown
        for remaining in range(config.HOLD_DURATION - 1, -1, -1):
            decision_locked = controller.decide(0.10, 0.05)
            self.assertEqual(decision_locked, 'HOLD', f"Controller failed to maintain HOLD countdown at remaining={remaining}")
            self.assertEqual(controller.hold_countdown, remaining)

if __name__ == '__main__':
    unittest.main()
