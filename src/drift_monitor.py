from scipy import stats
import numpy as np
from src import config

class DriftMonitor:
    def __init__(self, reference_data=None, window_size=None, p_value_threshold=None):
        """
        Tracks drift by comparing the current window to the immediately preceding window.
        reference_data is kept in signature for compatibility but is no longer used for drift.
        """
        self.window_size = window_size if window_size is not None else config.WINDOW_SIZE
        # We need a buffer of size 2 * window_size to compare previous and current windows
        self.buffer_size = 2 * self.window_size
        self.stream_buffer = []
        
    def update_and_check(self, x):
        """
        Adds a new observation to the buffer and checks for drift.
        x: numpy array of shape (n_features,)
        Returns the maximum KS statistic across all features (drift score)
        between the current window and the previous window.
        """
        self.stream_buffer.append(x)
        if len(self.stream_buffer) > self.buffer_size:
            self.stream_buffer.pop(0)
            
        if len(self.stream_buffer) < self.buffer_size:
            return 0.0
            
        # Split buffer into previous and current windows
        buffer_np = np.array(self.stream_buffer)
        prev_window = buffer_np[:self.window_size]
        curr_window = buffer_np[self.window_size:]
        
        max_stat = 0.0
        n_features = curr_window.shape[1]
        
        for f in range(n_features):
            ref_feat = prev_window[:, f]
            cur_feat = curr_window[:, f]
            
            # KS Test: null hypothesis is that the two samples are drawn from the same distribution
            stat, p_val = stats.ks_2samp(ref_feat, cur_feat)
            
            if stat > max_stat:
                max_stat = stat
                
        # Drift score is the maximum distance between empirical CDFs of the two sequential windows
        return max_stat
