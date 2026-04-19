import time
import numpy as np
from collections import deque
from scipy.signal import butter, filtfilt, find_peaks
import pylsl

class EEGStream:
    def __init__(self):
        streams = pylsl.resolve_byprop("name", "WS-default", timeout=10)
        self.inlet = pylsl.StreamInlet(streams[0])
        self.fs = int(self.inlet.info().nominal_srate())
        info = self.inlet.info()
        desc = info.desc()

        channels = desc.child("channels").child("channel")

        self.ch_names = []
        for _ in range(info.channel_count()):
            self.ch_names.append(channels.child_value("label"))
            channels = channels.next_sibling()

      
    def pull(self):
        chunk, timestamps = self.inlet.pull_chunk(timeout=0.0)
        return np.array(chunk), np.array(timestamps)