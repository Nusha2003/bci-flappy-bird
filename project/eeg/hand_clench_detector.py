import time
from pathlib import Path

import joblib
import numpy as np
import pylsl

from model.decode import DEFAULT_DATA_DIR, DEFAULT_MODEL_PATH, train_model


class HandClenchDetector:
    def __init__(self, fs, model_path: str | Path | None = None):
        self.fs = fs
        self.model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH

        self.recent_clenches = []
        self.refractory = 0.35
        self.last_clench_time = 0
        self.merge_tol = 0.06

        self.threshold = 0.5
        self.calibrated = False

        artifact = self._load_or_train_model()
        self.pipeline = artifact["pipeline"]
        self.window_samples = int(artifact["window_samples"])

    def _load_or_train_model(self):
        if self.model_path.exists():
            return joblib.load(self.model_path)

        artifact, _, _, _, _ = train_model(
            data_path=DEFAULT_DATA_DIR,
            output_model_path=self.model_path,
        )
        return artifact

    def _prepare_epoch(self, signal):
        x = np.asarray(signal, dtype=float)
        if x.ndim == 1:
            x = x[:, np.newaxis]

        if x.ndim != 2:
            raise ValueError(f"Expected 2D signal, got shape {x.shape}")

        if x.shape[0] < x.shape[1]:
            samples_first = x.T
        else:
            samples_first = x

        if samples_first.shape[0] < self.window_samples:
            return None

        window = samples_first[-self.window_samples :, :]
        return window.T[np.newaxis, :, :]

    def _predict_probability(self, signal):
        epoch = self._prepare_epoch(signal)
        if epoch is None:
            return None

        return float(self.pipeline.predict_proba(epoch)[0, 1])

    def calibrate(self, clench_signal, rest_signal):
        clench_probs = self._score_calibration_signal(clench_signal)
        rest_probs = self._score_calibration_signal(rest_signal)

        if len(clench_probs) == 0 or len(rest_probs) == 0:
            print("Calibration failed: not enough hand clench calibration windows")
            return False

        clench_floor = float(np.percentile(clench_probs, 25))
        rest_ceiling = float(np.percentile(rest_probs, 90))
        self.threshold = max(0.5, (clench_floor + rest_ceiling) / 2.0)
        self.calibrated = True

        print(f"Calibration complete. Probability threshold = {self.threshold:.3f}")
        return True

    def _score_calibration_signal(self, signal):
        x = np.asarray(signal, dtype=float)
        if x.ndim == 1:
            x = x[:, np.newaxis]
        if x.shape[0] < x.shape[1]:
            x = x.T

        if x.shape[0] < self.window_samples:
            return []

        step = max(self.window_samples // 2, 1)
        probs = []
        for end in range(self.window_samples, x.shape[0] + 1, step):
            window = x[end - self.window_samples : end, :]
            epoch = window.T[np.newaxis, :, :]
            probs.append(float(self.pipeline.predict_proba(epoch)[0, 1]))
        return probs

    def detect(self, signal, timestamps, thresh_min):
        timestamps = np.asarray(timestamps, dtype=float)

        prob = self._predict_probability(signal)
        if prob is None:
            return np.array([], dtype=int), [], {
                "threshold": float(self.threshold),
                "probability": 0.0,
            }

        threshold = max(float(self.threshold), float(thresh_min))
        peaks = np.array([], dtype=int)
        new_clenches = []
        now = time.time()

        if prob >= threshold and len(timestamps) > 0:
            event_time = float(timestamps[-1])

            if not any(abs(event_time - prev) < self.merge_tol for prev in self.recent_clenches):
                if now - self.last_clench_time >= self.refractory:
                    self.last_clench_time = now
                    self.recent_clenches.append(event_time)
                    new_clenches.append(event_time)

        now_lsl = pylsl.local_clock()
        self.recent_clenches = [t for t in self.recent_clenches if now_lsl - t < 2]

        info = {
            "threshold": float(threshold),
            "probability": float(prob),
        }
        return peaks, new_clenches, info
