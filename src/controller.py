class SUSController:
    def __init__(self, anomaly_threshold=0.58, drift_threshold=0.75, window_size=50):
        """
        Initializes the SUS Controller with calibrated thresholds.
        anomaly_threshold: raw score threshold
        drift_threshold: KS statistic threshold
        window_size: sliding window for aggregating anomaly evidence
        """
        self.anomaly_threshold = anomaly_threshold
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        self.anomaly_buffer = []
        self.consecutive_adapt = 0
        self.hold_countdown = 0
        
        # Pull from config if available, otherwise default
        from src import config
        self.min_consecutive = getattr(config, 'MIN_ADAPTATION_EVIDENCE', 3)
        self.hold_duration = getattr(config, 'HOLD_DURATION', 10)
        
    def decide(self, anomaly_score, drift_score, state=None):
        """
        Implements the 4-quadrant policy to select an adaptation action.
        Returns one of: 'KEEP', 'ADAPT', 'HOLD'
        """
        self.anomaly_buffer.append(anomaly_score)
        if len(self.anomaly_buffer) > self.window_size:
            self.anomaly_buffer.pop(0)
            
        if len(self.anomaly_buffer) < self.window_size:
            return None
            
        # FIX 2: Windowed aggregate (rolling fraction > threshold)
        fraction_anomalous = sum(1 for s in self.anomaly_buffer if s > self.anomaly_threshold) / self.window_size
        
        # High anomaly evidence if more than 10% of recent points exceed the strict threshold
        is_high_anomaly = fraction_anomalous > 0.10
        is_high_drift = drift_score > self.drift_threshold
        
        raw_decision = 'KEEP'
        # Quadrant 1: Normal data
        if not is_high_drift and not is_high_anomaly:
            raw_decision = 'KEEP'
            
        # Quadrant 2: Spike anomaly without statistical shift
        elif not is_high_drift and is_high_anomaly:
            raw_decision = 'KEEP'
            
        # Quadrant 3: Distribution shift without extreme anomalies
        elif is_high_drift and not is_high_anomaly:
            raw_decision = 'ADAPT'
            
        # Quadrant 4: Massive shift + massive anomaly (persistent contamination suspected)
        elif is_high_drift and is_high_anomaly:
            raw_decision = 'HOLD'
            
        # Apply Safeguards
        if self.hold_countdown > 0:
            self.hold_countdown -= 1
            self.consecutive_adapt = 0
            return 'HOLD'
            
        if raw_decision == 'HOLD':
            self.hold_countdown = self.hold_duration
            self.consecutive_adapt = 0
            return 'HOLD'
            
        if raw_decision == 'ADAPT':
            self.consecutive_adapt += 1
            if self.consecutive_adapt >= self.min_consecutive:
                return 'ADAPT'
            else:
                return 'KEEP' # Wait for more evidence
                
        if raw_decision == 'KEEP':
            self.consecutive_adapt = 0
            return 'KEEP'
            
        return 'KEEP'
