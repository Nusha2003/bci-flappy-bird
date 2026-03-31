import time
import json
import numpy as np
import zmq
from collections import deque
from mne_lsl.stream import StreamLSL

class BlinkDetector:
    def __init__(self, sfreq, threshold=150.0):
        self.sfreq = sfreq
        self.threshold = threshold

        self.W_widths = [int(0.05 * sfreq), int(0.1 * sfreq), int(0.15 * sfreq)]
        self.msdw_history = deque([0.0, 0.0, 0.0], maxlen=3)
        self.last_flap_time = 0
        self.min_cooldown = 0.15  # 150ms minimum between physical blinks

    def calculate_msdw(self, data_window):
        """: F(t) = |S(t) - S(t-W)|"""
        if data_window.shape[-1] < max(self.W_widths):
            return 0.0
        
        current_sample = data_window[-1]

        f_ws = [np.abs(current_sample - data_window[-w]) for w in self.W_widths]
        return max(f_ws)

    def is_local_peak(self):
        h = self.msdw_history
        return h[1] > h[0] and h[1] > h[2] and h[1] > self.threshold

def main():
    try:
        stream = StreamLSL(name='OpenBCI').connect()
        stream.filter(1.0, 15.0, picks="eeg") # Blink sweet spot
        sfreq = stream.info['sfreq']
    except Exception as e:
        print(f"Error connecting to LSL: {e}")
        return

    # ZMQ Setup (Publisher)
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind("tcp://*:5556") 

    detector = BlinkDetector(sfreq=sfreq)
    
    print(f"BCI Online @ {sfreq}Hz. Ready to flap.")

    while True:
        chunk, ts = stream.pull_chunk()
        
        if chunk.size > 0:
            data_window = stream.get_data(winsize=0.5, picks="eeg")[0][0]
            msdw_val = detector.calculate_msdw(data_window)
            detector.msdw_history.append(msdw_val)

            now = time.time()
            if detector.is_local_peak():
                if (now - detector.last_flap_time) > detector.min_cooldown:
                    payload = {"action": "FLAP", "val": float(msdw_val), "t": now}
                    pub.send_string(f"GAME_CMD {json.dumps(payload)}")
                    
                    print(f"Blink! MSDW: {msdw_val:.2f}")
                    detector.last_flap_time = now
        time.sleep(1 / (sfreq * 2))

if __name__ == "__main__":
    main()
