#methods for handling the logic behind EEG signals contolling the game


import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from eeg.blink_detector import BlinkDetector
from eeg.jaw_clench_detector import JawClenchDetector
from eeg.preprocess import preprocess
from eeg.stream import EEGStream


@dataclass
class ControllerUpdate:
    rel_times: np.ndarray
    signal: np.ndarray
    status_text: str | None = None
    jump_now: bool = False


class EEGController:
    def __init__(self, mode: int):
        self.mode = mode
        self.stream = EEGStream()
        ch_list = ['Pz', 'F4', 'C4', 'P4', 'P3', 'C3', 'F3', 'TRG']
        self.pick_channels = ['F4']
        self.indices = [self.stream.ch_names.index(ch) for ch in self.pick_channels]


        #need to change this - just looking athe channel we are interested in


        #the length in time that the plot displays
        self.plot_duration = 2


        self.blink_thresh_min = 70.0
        self.jaw_thresh_min = 0.2
        self.blink_veto_thresh = 70.0

        self.buffer = deque(maxlen=int(self.stream.fs * 5))
        self.time_buffer = deque(maxlen=int(self.stream.fs * 5))

        self.event_count = 0

        self.calibrating = True
        self.calibration_data = []
        self.calibration_duration = 5
        self.calibration_start_time = None
        self.calib_dot_count = 0
        self.calib_detected = 0

        self.clench_hold_until = 0.0
        self.last_jump_time = 0.0
        self.jump_interval = 0.18
        self.hold_seconds = 0.35

        self.jaw_block_until = 0.0
        self.jaw_block_seconds = 0.25


        if self.mode == 1:
            self.detector = BlinkDetector(self.stream.fs)
            self.blink_veto_detector = None
            self.window_title = "EEG Eye Blink Flappy Bird"
        else:
            self.detector = JawClenchDetector(self.stream.fs)
            self.blink_veto_detector = BlinkDetector(self.stream.fs)
            self.window_title = "EEG Jaw Clench Flappy Bird"

        self.status_text = self._build_calibration_status()

    #UI stuff to track calibration
    def tick_calibration_indicator(self) -> str | None:
        if not self.calibrating:
            return None

        self.calib_dot_count = (self.calib_dot_count + 1) % 4
        self.status_text = self._build_calibration_status()
        return self.status_text

    #main code for processing the newest batch of EEG data
    #if jaw clench - implement veto eye blinks
    #if in calibrating mode - run calibration on the data
    #Controller update: runs UI ready state
    def process_eeg(self) -> ControllerUpdate | None:
        chunk, ts = self.stream.pull()
        if len(ts) == 0:
            return None

        channel_stream = chunk[:, self.indices[0]]

        for value, timestamp in zip(channel_stream, ts):
            self.buffer.append(value)
            self.time_buffer.append(timestamp)

        if len(self.buffer) < int(self.stream.fs):
            return None

        window_samples = int(self.stream.fs * self.plot_duration)
        data = np.array(self.buffer)[-window_samples:]
        times = np.array(self.time_buffer)[-window_samples:]
        signal = preprocess(data, self.stream.fs)

        blink_events = []
        if self.mode == 2 and self.blink_veto_detector is not None:
            _, blink_events = self.blink_veto_detector.detect(
                signal, times, self.blink_veto_thresh
            )

        if self.calibrating:
            self.calibration_data.extend(channel_stream.tolist())
            self._run_calibration(signal, times, blink_events)
            return ControllerUpdate(
                rel_times=times - times[-1],
                signal=signal,
                status_text=self.status_text,
            )

        if self.mode == 1:
            jump_now = self._handle_blink(signal, times)
        else:
            jump_now = self._handle_jaw(signal, times, blink_events)

        return ControllerUpdate(
            rel_times=times - times[-1],
            signal=signal,
            status_text=self.status_text,
            jump_now=jump_now,
        )

    #is the jump being held - if still in a jump state - returns True
    def consume_held_jump(self) -> bool:
        now = time.monotonic()
        if self.mode != 2 or self.calibrating:
            return False
        if now >= self.clench_hold_until:
            return False

        if (now - self.last_jump_time) < self.jump_interval:
            return False

        self.last_jump_time = now
        return True

    #calibration instruction
    def _calibration_instruction(self) -> str:
        if self.mode == 1:
            return "Blink once or twice when prompted; keep your jaw relaxed."
        return "Clench steadily when prompted; try not to blink during calibration."

    #
    def _build_calibration_status(self) -> str:
        dots = "." * self.calib_dot_count
        pad = " " * (3 - self.calib_dot_count)
        instruction = self._calibration_instruction()

        if self.mode == 1:
            return (
                f"Calibrating blinks{dots}{pad}  - blinks detected: "
                f"{self.calib_detected}  |  {instruction}"
            )

        return (
            f"Calibrating jaw clenches{dots}{pad}  - clenches detected: "
            f"{self.calib_detected}  |  {instruction}"
        )

    
    def _build_ready_status(self) -> str:
        instruction = self._calibration_instruction()
        if self.mode == 1:
            return f"Calibration complete! Blink to flap.  |  {instruction}"
        return f"Calibration complete! Clench to flap.  |  {instruction}"

    #run calibration -  starts calibration timer
    #counts calibration events
    #builds a final calibration signal
    #gives calibration to detector
    #marks calibration complete and caluclates personalized threshold

    def _run_calibration(
        self,
        signal: np.ndarray,
        times: np.ndarray,
        blink_events: list[float] | None = None,
    ) -> None:
        if self.calibration_start_time is None:
            self.calibration_start_time = times[-1]

        if self.mode == 1:
            _, events = self.detector.detect(signal, times, self.blink_thresh_min)
        else:
            if blink_events:
                events = []
            else:
                _, events, _ = self.detector.detect(signal, times, self.jaw_thresh_min)

        if events:
            self.calib_detected += len(events)
            self.status_text = self._build_calibration_status()

        elapsed = times[-1] - self.calibration_start_time
        if elapsed < self.calibration_duration:
            return

        calib_signal = preprocess(np.array(self.calibration_data), self.stream.fs)

        if hasattr(self.detector, "calibrate"):
            try:
                self.detector.calibrate(calib_signal)
            except Exception:
                pass

        self.calibrating = False
        self.calib_detected = 0
        self.status_text = self._build_ready_status()

    def _handle_blink(self, signal: np.ndarray, times: np.ndarray) -> bool:
        _, blinks = self.detector.detect(signal, times, self.blink_thresh_min)
        if not blinks:
            return False

        self.event_count += len(blinks)
        self.status_text = f"Blinks: {self.event_count}"
        return True

    def _handle_jaw(
        self,
        signal: np.ndarray,
        times: np.ndarray,
        blink_events: list[float] | None = None,
    ) -> bool:
        now = time.monotonic()

        if blink_events:
            self.jaw_block_until = now + self.jaw_block_seconds
            return False

        if now < self.jaw_block_until:
            return False

        _, clenches, _ = self.detector.detect(signal, times, self.jaw_thresh_min)
        if not clenches:
            return False

        self.event_count += len(clenches)
        self.status_text = f"Jaw clenches: {self.event_count}"
        self.clench_hold_until = now + self.hold_seconds
        return False
