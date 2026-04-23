import time
import numpy as np

from controller import ControllerUpdate


class FakeEEGController:
    """Drop-in controller that simulates EEG without an LSL stream."""

    def __init__(self, mode: int):
        self.mode = mode
        self.fs = 250
        self.plot_duration = 2.0
        self.plot_channel_name = "Simulated F4"

        self.blink_thresh_min = 70.0
        self.jaw_thresh_min = 0.2
        self.hand_thresh_min = 0.2
        self.blink_veto_thresh = 70.0

        self.event_count = 0
        self.calibrating = True
        self.calibration_duration = 8
        self.calib_dot_count = 0
        self.calib_detected = 0
        self.calibration_started_at = time.monotonic()

        self.clench_hold_until = 0.0
        self.last_jump_time = 0.0
        self.jump_interval = 0.18
        self.hold_seconds = 0.35

        self.window_title = {
            1: "Fake EEG Eye Blink Flappy Bird",
            2: "Fake EEG Jaw Clench Flappy Bird",
            3: "Fake EEG Hand Clench Flappy Bird",
        }.get(self.mode, "Fake EEG Flappy Bird")

        self._manual_event_times: list[float] = []
        self._last_auto_event = time.monotonic()
        self._auto_period = {1: 2.5, 2: 3.0, 3: 3.2}.get(self.mode, 3.0)
        self._pending_manual_jump = False
        self.status_text = self._build_calibration_status()

    def tick_calibration_indicator(self) -> str | None:
        if not self.calibrating:
            return None
        self.calib_dot_count = (self.calib_dot_count + 1) % 4
        self.status_text = self._build_calibration_status()
        return self.status_text

    def trigger_manual_event(self) -> None:
        now = time.monotonic()
        self._manual_event_times.append(now)
        self._manual_event_times = [t for t in self._manual_event_times if now - t < 2.0]

        if self.calibrating:
            self.calib_detected += 1
            self.status_text = self._build_calibration_status()
            return

        if self.mode == 1:
            self.event_count += 1
            self._pending_manual_jump = True
            self.status_text = (
                f"Fake blinks: {self.event_count}  |  Press SPACE to simulate another blink."
            )
        else:
            self.event_count += 1
            self.clench_hold_until = now + self.hold_seconds
            label = "jaw clenches" if self.mode == 2 else "hand clenches"
            self.status_text = (
                f"Fake {label}: {self.event_count}  |  Press SPACE to simulate another clench."
            )

    def process_eeg(self) -> ControllerUpdate | None:
        now = time.monotonic()
        signal = self._generate_signal(now)
        rel_times = np.linspace(-self.plot_duration, 0.0, signal.size)

        if self.calibrating:
            elapsed = now - self.calibration_started_at
            self.calib_detected = max(self.calib_detected, int(elapsed // 2))
            if elapsed >= self.calibration_duration:
                self.calibrating = False
                self.status_text = self._build_ready_status()

            return ControllerUpdate(
                rel_times=rel_times,
                signal=signal,
                status_text=self.status_text,
            )

        jump_now = False
        if self.mode == 1 and self._pending_manual_jump:
            jump_now = True
            self._pending_manual_jump = False
        if self.mode == 1 and self._should_auto_trigger(now):
            self.trigger_manual_event()
            jump_now = True

        return ControllerUpdate(
            rel_times=rel_times,
            signal=signal,
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

    def _should_auto_trigger(self, now: float) -> bool:
        if now - self._last_auto_event < self._auto_period:
            return False
        self._last_auto_event = now
        return True

    def _generate_signal(self, now: float) -> np.ndarray:
        samples = int(self.fs * self.plot_duration)
        t = np.linspace(now - self.plot_duration, now, samples)

        base = (
            18.0 * np.sin(2 * np.pi * 1.4 * t)
            + 9.0 * np.sin(2 * np.pi * 3.2 * t + 0.7)
            + 3.0 * np.sin(2 * np.pi * 8.0 * t + 1.1)
        )
        noise = np.random.normal(0.0, 4.0, size=samples)
        signal = base + noise

        active_events = [event_t for event_t in self._manual_event_times if now - event_t < 2.0]
        self._manual_event_times = active_events

        amplitude = {1: 85.0, 2: 60.0, 3: 55.0}.get(self.mode, 70.0)
        width = {1: 0.04, 2: 0.08, 3: 0.08}.get(self.mode, 0.06)
        for event_t in active_events:
            signal += amplitude * np.exp(-0.5 * ((t - event_t) / width) ** 2)

        if self.calibrating and self.mode != 1:
            phase = int((now - self.calibration_started_at) // 2) % 2
            if phase == 1:
                center = now - 0.25
                signal += amplitude * np.exp(-0.5 * ((t - center) / width) ** 2)

        return signal.astype(float)

    def _build_calibration_status(self) -> str:
        dots = "." * self.calib_dot_count
        pad = " " * (3 - self.calib_dot_count)
        if self.mode == 1:
            instruction = "Press SPACE to simulate a blink during demo calibration."
            return (
                f"Fake blink calibration{dots}{pad}  - demo events: "
                f"{self.calib_detected}  |  {instruction}"
            )
        if self.mode == 2:
            instruction = "Press SPACE to simulate a jaw clench during demo calibration."
            return (
                f"Fake jaw calibration{dots}{pad}  - demo events: "
                f"{self.calib_detected}  |  {instruction}"
            )
        instruction = "Press SPACE to simulate a hand clench during demo calibration."
        return (
            f"Fake hand calibration{dots}{pad}  - demo events: "
            f"{self.calib_detected}  |  {instruction}"
        )

    def _build_ready_status(self) -> str:
        if self.mode == 1:
            return "Fake calibration complete! Press SPACE to blink-flap."
        if self.mode == 2:
            return "Fake calibration complete! Press SPACE to jaw-clench flap."
        return "Fake calibration complete! Press SPACE to hand-clench flap."
