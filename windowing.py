from collections import deque
import numpy as np


#Rolling window, 20 frames because it gave best results.
class WindowBuffer:
    def __init__(self, size=20):
        self.size = size
        self.buffer = deque(maxlen=size)

    def add(self, x):
        self.buffer.append(x)

    def is_full(self):
        return len(self.buffer) == self.size

    def get_window(self):
        #shape = (window_size, feature_dim)
        return np.array(self.buffer)