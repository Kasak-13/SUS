class SUSController:
    def __init__(self, anomaly_threshold=None, drift_threshold=None, anomaly_fraction_threshold=None, window_size=None):
        """
        Initializes the SUS Controller with calibrated thresholds and safeguards.
        anomaly_threshold: raw score threshold (defaults to config.ANOMALY_THRESHOLD = 0.58)
        drift_threshold: KS statistic threshold (defaults to config.DRIFT_THRESHOLD = 0.75)
        anomaly_fraction_threshold: windowed extreme fraction (defaults to config.ANOMALY_FRACTION_THRESHOLD = 0.10)
        window_size: sliding window for aggregating anomaly evidence (defaults to config.WINDOW_SIZE = 50)
        """
        from src import config
        self.anomaly_threshold = anomaly_threshold if anomaly_threshold is not None else getattr(config, 'ANOMALY_THRESHOLD', 0.58)
        self.drift_threshold = drift_threshold if drift_threshold is not None else getattr(config, 'DRIFT_THRESHOLD', 0.75)
        self.anomaly_fraction_threshold = anomaly_fraction_threshold if anomaly_fraction_threshold is not None else getattr(config, 'ANOMALY_FRACTION_THRESHOLD', 0.10)
        self.window_size = window_size if window_size is not None else getattr(config, 'WINDOW_SIZE', 50)
        
        self.anomaly_buffer = []
        self.consecutive_adapt = 0
        self.hold_countdown = 0
        
        # Safeguards from config
        self.min_consecutive = getattr(config, 'MIN_CONSECUTIVE_OBSERVATIONS', 3)
        self.hold_duration = getattr(config, 'HOLD_DURATION', 10)
        
    def record_observation(self, anomaly_score):
        """
        Records an anomaly score into the rolling window buffer.
        Maintains the sliding window size constraint.
        """
        self.anomaly_buffer.append(anomaly_score)
        if len(self.anomaly_buffer) > self.window_size:
            self.anomaly_buffer.pop(0)
            
    def decide(self, anomaly_score, drift_score, state=None):
        """
        Implements the 4-quadrant policy to select an adaptation action.
        Returns one of: 'KEEP', 'ADAPT', 'HOLD'
        """
        self.record_observation(anomaly_score)
            
        if len(self.anomaly_buffer) < self.window_size:
            return None
            
        # Windowed aggregate (rolling fraction > threshold)
        fraction_anomalous = sum(1 for s in self.anomaly_buffer if s > self.anomaly_threshold) / self.window_size
        
        # High anomaly evidence if recent points exceeding strict threshold cross fraction threshold (0.10)
        is_high_anomaly = fraction_anomalous > self.anomaly_fraction_threshold
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
