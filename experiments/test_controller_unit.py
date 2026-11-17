import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.controller import SUSController
from src.sus_detector import SUSDetector

class TestSUSControllerUnit(unittest.TestCase):
    
    def test_centralized_configuration_import(self):
        """Verify that SUSController defaults match centralized src.config values."""
        controller = SUSController()
        
        self.assertEqual(controller.anomaly_threshold, config.ANOMALY_THRESHOLD)
        self.assertEqual(controller.anomaly_threshold, 0.58)
        
        self.assertEqual(controller.drift_threshold, config.DRIFT_THRESHOLD)
        self.assertEqual(controller.drift_threshold, 0.75)
        
        self.assertEqual(controller.anomaly_fraction_threshold, config.ANOMALY_FRACTION_THRESHOLD)
        self.assertEqual(controller.anomaly_fraction_threshold, 0.10)
        
        self.assertEqual(controller.min_consecutive, config.MIN_CONSECUTIVE_OBSERVATIONS)
        self.assertEqual(controller.min_consecutive, 3)
        
        self.assertEqual(controller.hold_duration, config.HOLD_DURATION)
        self.assertEqual(controller.hold_duration, 10)
        
        self.assertEqual(controller.window_size, config.WINDOW_SIZE)
        self.assertEqual(controller.window_size, 50)
        
    def test_warmup_buffering(self):
        """Verify controller returns None until buffer reaches window_size."""
        controller = SUSController()
        for i in range(config.WINDOW_SIZE - 1):
            decision = controller.decide(anomaly_score=0.2, drift_score=0.1)
            self.assertIsNone(decision, f"Step {i} should be buffering (None)")
            
        decision = controller.decide(anomaly_score=0.2, drift_score=0.1)
        self.assertEqual(decision, 'KEEP', "Step 50 should produce first decision 'KEEP'")

    def test_min_consecutive_observations_safeguard(self):
        """Verify MIN_CONSECUTIVE_OBSERVATIONS = 3 requires exactly 3 consecutive ADAPT proposals."""
        controller = SUSController()
        # Fill buffer with normal points
        for _ in range(config.WINDOW_SIZE):
            controller.decide(anomaly_score=0.2, drift_score=0.1)
            
        # Proposal 1: High drift (0.80 > 0.75), low anomaly (0.2 < 0.58)
        # Should be filtered by safeguard and return KEEP
        d1 = controller.decide(anomaly_score=0.2, drift_score=0.80)
        self.assertEqual(d1, 'KEEP', "First ADAPT proposal should be filtered by 3-consecutive safeguard")
        self.assertEqual(controller.consecutive_adapt, 1)
        
        # Proposal 2: Second consecutive
        d2 = controller.decide(anomaly_score=0.2, drift_score=0.80)
        self.assertEqual(d2, 'KEEP', "Second ADAPT proposal should be filtered by 3-consecutive safeguard")
        self.assertEqual(controller.consecutive_adapt, 2)
        
        # Proposal 3: Third consecutive -> GATE PASSES!
        d3 = controller.decide(anomaly_score=0.2, drift_score=0.80)
        self.assertEqual(d3, 'ADAPT', "Third consecutive proposal should trigger ADAPT")
        self.assertEqual(controller.consecutive_adapt, 3)

    def test_hold_duration_lockdown_safeguard(self):
        """Verify HOLD_DURATION = 10 locks controller in HOLD for 10 steps after alert."""
        controller = SUSController()
        # Fill buffer with high anomaly scores to exceed 0.10 fraction threshold
        # 6 out of 50 = 12% > 10%
        for _ in range(44):
            controller.decide(anomaly_score=0.2, drift_score=0.1)
        for _ in range(6):
            controller.decide(anomaly_score=0.75, drift_score=0.1)
            
        # Trigger HOLD with high drift (0.80 > 0.75) and high anomaly (>10% extreme fraction)
        d_hold = controller.decide(anomaly_score=0.75, drift_score=0.80)
        self.assertEqual(d_hold, 'HOLD', "Massive drift + dense anomaly must trigger HOLD")
        
        # Now feed normal low-drift inputs, verify that HOLD persists for hold_duration (10 steps)
        for step in range(10):
            d_locked = controller.decide(anomaly_score=0.1, drift_score=0.1)
            self.assertEqual(d_locked, 'HOLD', f"Lockdown step {step+1}/10 should remain HOLD")
            
    def test_sus_detector_configuration_alignment(self):
        """Verify SUSDetector initializes with config.ANOMALY_THRESHOLD."""
        detector = SUSDetector()
        self.assertEqual(detector.anomaly_threshold, config.ANOMALY_THRESHOLD)
        self.assertEqual(detector.controller.anomaly_threshold, config.ANOMALY_THRESHOLD)
        self.assertEqual(detector.controller.drift_threshold, config.DRIFT_THRESHOLD)

if __name__ == '__main__':
    unittest.main()
