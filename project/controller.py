import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from eeg.blink_detector import BlinkDetector
from eeg.hand_clench_detector import HandClenchDetector
from eeg.ml_jaw_clench_detector import MLJawClenchDetector
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

        self.plot_duration = 2

        self.blink_thresh_min = 70.0
        self.jaw_thresh_min = 0.2
        self.hand_thresh_min = 0.2
        self.blink_veto_thresh = 70.0

        self.buffer = deque(maxlen=int(self.stream.fs * 5))
        self.time_buffer = deque(maxlen=int(self.stream.fs * 5))

        self.event_count = 0

        self.calibrating = True
        self.calibration_data = []
        self.calibration_duration = 10
        self.calibration_start_time = None
        self.calib_dot_count = 0
        self.calib_detected = 0

        self.hand_calibration_target = 20
        self.hand_calibration_samples_per_phase = int(self.stream.fs * 3.0)
        self.hand_calibration_phase = "rest"
        self.hand_phase_buffer = []
        self.hand_rest_segments = []
        self.hand_clench_segments = []

        self.jaw_calibration_target = 5
        self.jaw_calibration_prepare_seconds = 1.5
        self.jaw_calibration_hold_seconds = 3.0
        self.jaw_calibration_break_seconds = 1.0
        self.jaw_calibration_phase = "prepare"
        self.jaw_phase_started_at = None
        self.jaw_trial_index = 0
        self.jaw_trial_order = []
        self.jaw_trial_buffer = []
        self.jaw_calibration_blocks = []
        self.jaw_calibration_labels = []

        self.clench_hold_until = 0.0
        self.last_jump_time = 0.0
        self.jump_interval = 0.18
        self.hold_seconds = 0.35

        self.jaw_block_until = 0.0
        self.jaw_block_seconds = 0.25

        self.signal_channel_indices = self._resolve_channel_indices(
            ["F4", "C4", "P4", "P3", "C3", "F3"]
        )
        if not self.signal_channel_indices:
            self.signal_channel_indices = self._fallback_signal_indices()

        self.plot_channel_index = self._resolve_first_channel(
            ["F4", "F3", "C4", "C3"], fallback=self.signal_channel_indices[0]
        )
        self.plot_channel_name = self.stream.ch_names[self.plot_channel_index]

        self.blink_channel_indices = self._resolve_channel_indices(
            ["F4"]
        )
        if not self.blink_channel_indices:
            self.blink_channel_indices = [self.plot_channel_index]

        self.jaw_channel_indices = self._resolve_channel_indices(
            ["F4", "F3", "C4", "C3"]
        )
        if not self.jaw_channel_indices:
            self.jaw_channel_indices = [self.plot_channel_index]

        self.hand_channel_indices = self._resolve_channel_indices(
            ["F4", "C4", "P4", "P3", "C3", "F3"]
        )
        if not self.hand_channel_indices:
            self.hand_channel_indices = self.signal_channel_indices

        if self.mode == 1:
            self.detector = BlinkDetector(self.stream.fs)
            self.blink_veto_detector = None
            self.window_title = "EEG Eye Blink Flappy Bird"
            self.control_channel_indices = self.blink_channel_indices
        elif self.mode == 2:
            jaw_names = [self.stream.ch_names[idx] for idx in self.jaw_channel_indices]
            self.detector = MLJawClenchDetector(self.stream.fs, channel_names=jaw_names)
            self.blink_veto_detector = BlinkDetector(self.stream.fs)
            self.window_title = "EEG Jaw Clench Flappy Bird"
            self.control_channel_indices = self.jaw_channel_indices
            self._reset_jaw_calibration_trials()
        else:
            self.detector = HandClenchDetector(self.stream.fs)
            self.blink_veto_detector = None
            self.window_title = "EEG Hand Clench Flappy Bird"
            self.control_channel_indices = self.hand_channel_indices

        self.status_text = self._build_calibration_status()

    def tick_calibration_indicator(self) -> str | None:
        if not self.calibrating:
            return None

        self.calib_dot_count = (self.calib_dot_count + 1) % 4
        self.status_text = self._build_calibration_status()
        return self.status_text

    def process_eeg(self) -> ControllerUpdate | None:
        chunk, ts = self.stream.pull()
        if len(ts) == 0:
            return None

        for sample, timestamp in zip(chunk, ts):
            self.buffer.append(sample)
            self.time_buffer.append(timestamp)

        if self.mode == 2 and self.calibrating:
            self._run_jaw_calibration(chunk[:, self.control_channel_indices], ts)

        if self.mode == 3 and self.calibrating:
            self._run_hand_calibration(chunk[:, self.control_channel_indices])

        if len(self.buffer) < int(self.stream.fs):
            return None

        window_samples = int(self.stream.fs * self.plot_duration)
        data = np.array(self.buffer)[-window_samples:]
        times = np.array(self.time_buffer)[-window_samples:]

        plot_signal = preprocess(data[:, self.plot_channel_index], self.stream.fs)
        blink_signal = self._preprocess_mean(data[:, self.blink_channel_indices])
        control_signal = data[:, self.control_channel_indices]

        blink_events = []
        if self.mode == 2 and self.blink_veto_detector is not None:
            _, blink_events = self.blink_veto_detector.detect(
                blink_signal, times, self.blink_veto_thresh
            )

        if self.calibrating:
            if self.mode == 1:
                self.calibration_data.extend(blink_signal.tolist())
                self._run_single_channel_calibration(blink_signal, times)

            return ControllerUpdate(
                rel_times=times - times[-1],
                signal=plot_signal,
                status_text=self.status_text,
            )

        if self.mode == 1:
            jump_now = self._handle_blink(blink_signal, times)
        elif self.mode == 2:
            jump_now = self._handle_jaw(control_signal, times, blink_events)
        else:
            jump_now = self._handle_hand(control_signal, times)

        return ControllerUpdate(
            rel_times=times - times[-1],
            signal=plot_signal,
            status_text=self.status_text,
            jump_now=jump_now,
        )

    def consume_held_jump(self) -> bool:
        now = time.monotonic()
        if self.mode not in (2, 3) or self.calibrating:
            return False
        if now >= self.clench_hold_until:
            return False
        if (now - self.last_jump_time) < self.jump_interval:
            return False

        self.last_jump_time = now
        return True

    def _calibration_instruction(self) -> str:
        if self.mode == 1:
            return "Blink once or twice when prompted; keep your jaw relaxed."
        if self.mode == 2:
            return "Clench steadily when prompted; try not to blink during calibration."
        if self.hand_calibration_phase == "rest":
            return "REST: relax your hand and stay still."
        return "CLENCH: make a firm hand clench."

    def _build_calibration_status(self) -> str:
        dots = "." * self.calib_dot_count
        pad = " " * (3 - self.calib_dot_count)
        instruction = self._calibration_instruction()

        if self.mode == 1:
            return (
                f"Calibrating blinks{dots}{pad}  - blinks detected: "
                f"{self.calib_detected}  |  {instruction}"
            )

        if self.mode == 2:
            total = self.jaw_calibration_target * 2
            done = len(self.jaw_calibration_labels)
            rest_done = sum(label == 0 for label in self.jaw_calibration_labels)
            clench_done = sum(label == 1 for label in self.jaw_calibration_labels)
            cue = self._current_jaw_trial_name()
            phase_map = {
                "prepare": "get ready",
                "collect": "hold steady",
                "break": "relax",
                "train": "training model",
            }
            phase_text = phase_map.get(self.jaw_calibration_phase, self.jaw_calibration_phase)
            return (
                f"Jaw calibration{dots}{pad}  - trial {min(done + 1, total)}/{total}  |  "
                f"REST {rest_done}/{self.jaw_calibration_target}  |  "
                f"CLENCH {clench_done}/{self.jaw_calibration_target}  |  "
                f"{cue}: {phase_text}"
            )

        rest_done = len(self.hand_rest_segments)
        clench_done = len(self.hand_clench_segments)
        cue = "REST" if self.hand_calibration_phase == "rest" else "CLENCH"
        return (
            f"Hand calibration{dots}{pad}  - cue: {cue}  |  "
            f"rest {rest_done}/{self.hand_calibration_target}  |  "
            f"clench {clench_done}/{self.hand_calibration_target}  |  {instruction}"
        )

    def _build_ready_status(self) -> str:
        if self.mode == 1:
            return "Calibration complete! Blink to flap.  |  Blink naturally to jump."
        if self.mode == 2:
            train_acc = getattr(self.detector, "last_train_accuracy", 0.0)
            return (
                "Calibration complete! Clench to flap.  |  "
                f"Jaw model train acc: {train_acc:.2f}"
            )
        return (
            "Calibration complete! Hand clench to flap.  |  "
            "Use firm hand clenches to jump."
        )

    def _run_single_channel_calibration(
        self,
        signal: np.ndarray,
        times: np.ndarray,
    ) -> None:
        if self.calibration_start_time is None:
            self.calibration_start_time = times[-1]

        _, events = self.detector.detect(signal, times, self.blink_thresh_min)
        if events:
            self.calib_detected += len(events)
            self.status_text = self._build_calibration_status()

        if (times[-1] - self.calibration_start_time) < self.calibration_duration:
            return

        calib_signal = preprocess(np.array(self.calibration_data), self.stream.fs)
        success = True
        if hasattr(self.detector, "calibrate"):
            try:
                success = bool(self.detector.calibrate(calib_signal))
            except Exception:
                success = False

        if success:
            self.calibrating = False
            self.calib_detected = 0
            self.status_text = self._build_ready_status()
        else:
            self.calibration_data = []
            self.calibration_start_time = times[-1]
            self.calib_detected = 0
            self.status_text = "Blink calibration failed. Try again with clearer blinks."

    def _run_jaw_calibration(
        self,
        chunk: np.ndarray,
        timestamps: np.ndarray,
    ) -> None:
        if len(timestamps) == 0:
            return

        now_ts = float(timestamps[-1])
        if self.jaw_phase_started_at is None:
            self.jaw_phase_started_at = now_ts
            self.status_text = self._build_calibration_status()

        if self.jaw_calibration_phase == "prepare":
            if (now_ts - self.jaw_phase_started_at) >= self.jaw_calibration_prepare_seconds:
                self.jaw_calibration_phase = "collect"
                self.jaw_phase_started_at = now_ts
                self.jaw_trial_buffer = []
            self.status_text = self._build_calibration_status()
            return

        if self.jaw_calibration_phase == "collect":
            x_chunk = np.asarray(chunk, dtype=float)
            if x_chunk.ndim == 2 and x_chunk.shape[0] > 0:
                self.jaw_trial_buffer.append(x_chunk.copy())

            if (now_ts - self.jaw_phase_started_at) < self.jaw_calibration_hold_seconds:
                self.status_text = self._build_calibration_status()
                return

            if self.jaw_trial_buffer:
                block = np.concatenate(self.jaw_trial_buffer, axis=0)
                self.jaw_calibration_blocks.append(block)
                self.jaw_calibration_labels.append(self.jaw_trial_order[self.jaw_trial_index])

            self.jaw_trial_index += 1
            self.jaw_trial_buffer = []
            self.jaw_calibration_phase = "break"
            self.jaw_phase_started_at = now_ts
            self.status_text = self._build_calibration_status()
            return

        if self.jaw_calibration_phase == "break":
            if self.jaw_trial_index >= len(self.jaw_trial_order):
                self.jaw_calibration_phase = "train"
                self.status_text = self._build_calibration_status()
                success = False
                if hasattr(self.detector, "calibrate"):
                    try:
                        success = bool(
                            self.detector.calibrate(
                                self.jaw_calibration_blocks,
                                self.jaw_calibration_labels,
                            )
                        )
                    except Exception:
                        success = False

                if success:
                    self.calibrating = False
                    self.status_text = self._build_ready_status()
                else:
                    self._reset_jaw_calibration_trials()
                    self.status_text = (
                        "Jaw calibration failed. Stay still during REST and clench more firmly."
                    )
                return

            if (now_ts - self.jaw_phase_started_at) >= self.jaw_calibration_break_seconds:
                self.jaw_calibration_phase = "prepare"
                self.jaw_phase_started_at = now_ts

            self.status_text = self._build_calibration_status()
            return

    def _run_hand_calibration(self, chunk: np.ndarray) -> None:
        for sample in chunk:
            self.hand_phase_buffer.append(sample)
            if len(self.hand_phase_buffer) < self.hand_calibration_samples_per_phase:
                continue

            segment = np.asarray(self.hand_phase_buffer, dtype=float)
            self.hand_phase_buffer = []

            if self.hand_calibration_phase == "rest":
                self.hand_rest_segments.append(segment)
                self.hand_calibration_phase = "clench"
            else:
                self.hand_clench_segments.append(segment)
                self.hand_calibration_phase = "rest"

            self.status_text = self._build_calibration_status()

            if len(self.hand_rest_segments) >= self.hand_calibration_target and len(
                self.hand_clench_segments
            ) >= self.hand_calibration_target:
                break

        if len(self.hand_rest_segments) < self.hand_calibration_target:
            return
        if len(self.hand_clench_segments) < self.hand_calibration_target:
            return

        rest_signal = np.concatenate(self.hand_rest_segments, axis=0)
        clench_signal = np.concatenate(self.hand_clench_segments, axis=0)

        if hasattr(self.detector, "calibrate"):
            try:
                self.detector.calibrate(clench_signal, rest_signal)
            except Exception:
                pass

        self.calibrating = False
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

        _, clenches, info = self.detector.detect(signal, times, self.jaw_thresh_min)
        if not clenches:
            if isinstance(info, dict):
                prob = float(info.get("probability", 0.0))
                self.status_text = f"Jaw ready  |  clench prob: {prob:.2f}"
            return False

        self.event_count += len(clenches)
        prob = 0.0
        if isinstance(info, dict):
            prob = float(info.get("probability", 0.0))
        self.status_text = f"Jaw clenches: {self.event_count}  |  clench prob: {prob:.2f}"
        self.clench_hold_until = now + self.hold_seconds
        return False

    def _handle_hand(self, signal: np.ndarray, times: np.ndarray) -> bool:
        now = time.monotonic()

        _, clenches, _ = self.detector.detect(signal, times, self.hand_thresh_min)
        if not clenches:
            return False

        self.event_count += len(clenches)
        self.status_text = f"Hand clenches: {self.event_count}"
        self.clench_hold_until = now + self.hold_seconds
        return False

    def _preprocess_mean(self, signal: np.ndarray) -> np.ndarray:
        x = np.asarray(signal, dtype=float)
        if x.ndim == 1:
            return preprocess(x, self.stream.fs)

        channels = [preprocess(x[:, idx], self.stream.fs) for idx in range(x.shape[1])]
        return np.mean(np.column_stack(channels), axis=1)

    def _combine_channels(self, signal: np.ndarray) -> np.ndarray:
        x = np.asarray(signal, dtype=float)
        if x.ndim == 1:
            return x
        return np.mean(x, axis=1)

    def _reset_jaw_calibration_trials(self) -> None:
        self.jaw_trial_order = [0] * self.jaw_calibration_target + [1] * self.jaw_calibration_target
        np.random.default_rng().shuffle(self.jaw_trial_order)
        self.jaw_calibration_phase = "prepare"
        self.jaw_phase_started_at = None
        self.jaw_trial_index = 0
        self.jaw_trial_buffer = []
        self.jaw_calibration_blocks = []
        self.jaw_calibration_labels = []

    def _current_jaw_trial_name(self) -> str:
        if not self.jaw_trial_order:
            return "REST"
        idx = min(self.jaw_trial_index, len(self.jaw_trial_order) - 1)
        return "JAW CLENCH" if self.jaw_trial_order[idx] == 1 else "REST"

    def _resolve_channel_indices(self, desired_names: list[str]) -> list[int]:
        normalized = {
            self._normalize_name(name): idx
            for idx, name in enumerate(self.stream.ch_names)
        }

        indices = []
        for name in desired_names:
            idx = normalized.get(self._normalize_name(name))
            if idx is not None and idx not in indices:
                indices.append(idx)
        return indices

    def _resolve_first_channel(self, desired_names: list[str], fallback: int) -> int:
        matches = self._resolve_channel_indices(desired_names)
        if matches:
            return matches[0]
        return fallback

    def _fallback_signal_indices(self) -> list[int]:
        exclude = {"TRG", "TRIGGER", "EVENT"}
        indices = []
        for idx, name in enumerate(self.stream.ch_names):
            if self._normalize_name(name) in exclude:
                continue
            indices.append(idx)
        return indices[: max(1, min(6, len(indices)))]

    def _normalize_name(self, name: str) -> str:
        return "".join(ch for ch in name.upper() if ch.isalnum())
