from __future__ import annotations

import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def canonicalize_channel_name(name: str) -> str:
    return "".join(ch for ch in str(name).upper() if ch.isalnum())


def select_jaw_channel_indices(ch_names: list[str]) -> list[int]:
    """Pick frontal-priority channels that best capture jaw activity."""
    priority = {"FP1", "FP2", "AF3", "AF4", "F7", "F8", "F3", "F4"}
    idxs = [
        i for i, name in enumerate(ch_names)
        if canonicalize_channel_name(name) in priority
    ]
    if idxs:
        return idxs
    return list(range(len(ch_names)))


def extract_jaw_features(block: np.ndarray, jaw_idxs: list[int]) -> np.ndarray:
    """Extract a compact 12D jaw-clench feature vector from EEG samples."""
    x = np.asarray(block, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected shape (n_channels, n_samples), got {x.shape}")
    if x.shape[1] < 2:
        return np.zeros(12, dtype=np.float32)

    if jaw_idxs:
        x = x[jaw_idxs]

    x = x - np.mean(x, axis=1, keepdims=True)
    ptp = np.ptp(x, axis=1)
    rms = np.sqrt(np.mean(np.square(x), axis=1))
    mav = np.mean(np.abs(x), axis=1)
    line_len = np.mean(np.abs(np.diff(x, axis=1)), axis=1)

    def _summary(v: np.ndarray) -> list[float]:
        return [float(np.mean(v)), float(np.max(v)), float(np.std(v))]

    features = _summary(ptp) + _summary(rms) + _summary(mav) + _summary(line_len)
    return np.asarray(features, dtype=np.float32)


def split_overlapping_windows(
    block: np.ndarray,
    sfreq: float,
    window_s: float,
    step_s: float,
) -> np.ndarray:
    """Split a continuous block into overlapping windows."""
    x = np.asarray(block, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected shape (n_channels, n_samples), got {x.shape}")

    window_n = int(round(float(window_s) * float(sfreq)))
    step_n = int(round(float(step_s) * float(sfreq)))
    if window_n <= 0 or step_n <= 0:
        raise ValueError("window_s and step_s must produce at least one sample")
    if x.shape[1] < window_n:
        return np.empty((0, x.shape[0], window_n), dtype=np.float32)

    starts = np.arange(0, x.shape[1] - window_n + 1, step_n, dtype=int)
    windows = np.empty((len(starts), x.shape[0], window_n), dtype=np.float32)
    for i, start in enumerate(starts):
        windows[i] = x[:, start:start + window_n]
    return windows


def prepare_jaw_calibration_features(
    blocks: list[np.ndarray],
    labels: list[int],
    jaw_idxs: list[int],
    sfreq: float,
    window_s: float,
    step_s: float,
    edge_trim_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert long calibration blocks into windowed jaw feature rows."""
    if len(blocks) != len(labels):
        raise ValueError("blocks and labels must have the same length")

    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    trim_n = max(0, int(round(float(edge_trim_s) * float(sfreq))))

    for block, label in zip(blocks, labels):
        x_block = np.asarray(block, dtype=np.float32)
        if x_block.ndim != 2 or x_block.shape[1] == 0:
            continue

        start = trim_n
        stop = x_block.shape[1] - trim_n
        if stop <= start:
            continue

        trimmed = x_block[:, start:stop]
        windows = split_overlapping_windows(trimmed, sfreq, window_s, step_s)
        for win in windows:
            x_rows.append(extract_jaw_features(win, jaw_idxs))
            y_rows.append(int(label))

    if not x_rows:
        return np.empty((0, 12), dtype=np.float32), np.empty((0,), dtype=int)
    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=int)


def build_jaw_clench_classifier(
    *,
    random_state: int = 42,
    class_weight: str | None = "balanced",
    solver: str = "liblinear",
    max_iter: int = 1000,
) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                random_state=int(random_state),
                class_weight=class_weight,
                solver=str(solver),
                max_iter=int(max_iter),
            ),
        ),
    ])


class MLJawClenchDetector:
    """Windowed ML detector for jaw-clench classification."""

    def __init__(
        self,
        fs: float,
        channel_names: list[str] | None = None,
        window_s: float = 0.60,
        step_s: float = 0.10,
        edge_trim_s: float = 0.5,
        prob_threshold: float = 0.65,
    ) -> None:
        self.fs = float(fs)
        self.channel_names = list(channel_names or [])
        self.window_s = float(window_s)
        self.step_s = float(step_s)
        self.edge_trim_s = float(edge_trim_s)
        self.window_n = max(1, int(round(self.window_s * self.fs)))
        self.prob_threshold = float(prob_threshold)

        self.refractory = 0.5
        self.last_clench_time = 0.0
        self.prev_pred = 0

        self.classifier: Pipeline | None = None
        self.jaw_idxs = select_jaw_channel_indices(self.channel_names)
        self.calibrated = False
        self.last_train_accuracy = 0.0
        self.last_class_counts = {0: 0, 1: 0}

    def _to_channels_first(self, signal: np.ndarray) -> np.ndarray:
        x = np.asarray(signal, dtype=np.float32)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        elif x.ndim != 2:
            raise ValueError(f"Expected 1D or 2D signal, got shape {x.shape}")

        if x.shape[0] > x.shape[1]:
            x = x.T
        return x

    def calibrate(self, blocks: list[np.ndarray], labels: list[int]) -> bool:
        x_cal, y_cal = prepare_jaw_calibration_features(
            blocks=[self._to_channels_first(block) for block in blocks],
            labels=labels,
            jaw_idxs=self.jaw_idxs,
            sfreq=self.fs,
            window_s=self.window_s,
            step_s=self.step_s,
            edge_trim_s=self.edge_trim_s,
        )

        if x_cal.shape[0] < 12:
            print("Jaw calibration failed: not enough usable calibration windows")
            return False

        present = np.unique(y_cal)
        if present.size < 2:
            print("Jaw calibration failed: missing rest or clench examples")
            return False

        clf = build_jaw_clench_classifier()
        clf.fit(x_cal, y_cal)

        self.classifier = clf
        self.calibrated = True
        self.prev_pred = 0
        self.last_train_accuracy = float(clf.score(x_cal, y_cal))

        vals, cnts = np.unique(y_cal, return_counts=True)
        self.last_class_counts = {int(v): int(c) for v, c in zip(vals, cnts)}
        print(
            "Jaw calibration complete. "
            f"train_acc={self.last_train_accuracy:.3f}, "
            f"rest_windows={self.last_class_counts.get(0, 0)}, "
            f"clench_windows={self.last_class_counts.get(1, 0)}"
        )
        return True

    def detect(self, signal: np.ndarray, timestamps: np.ndarray, thresh_min: float):
        del thresh_min

        timestamps = np.asarray(timestamps, dtype=float)
        x = self._to_channels_first(signal)

        if timestamps.size == 0 or x.shape[1] < self.window_n:
            return np.array([], dtype=int), [], {
                "probability": 0.0,
                "threshold": self.prob_threshold,
                "predicted_class": 0,
            }

        if self.classifier is None:
            return np.array([], dtype=int), [], {
                "probability": 0.0,
                "threshold": self.prob_threshold,
                "predicted_class": 0,
            }

        window = x[:, -self.window_n:]
        feat = extract_jaw_features(window, self.jaw_idxs).reshape(1, -1)
        prob = float(self.classifier.predict_proba(feat)[0, 1])
        pred = int(prob >= self.prob_threshold)

        new_clenches: list[float] = []
        now = time.monotonic()
        if pred == 1 and self.prev_pred == 0 and (now - self.last_clench_time) >= self.refractory:
            self.last_clench_time = now
            new_clenches.append(float(timestamps[-1]))

        self.prev_pred = pred

        info = {
            "probability": prob,
            "threshold": self.prob_threshold,
            "predicted_class": pred,
            "train_accuracy": self.last_train_accuracy,
        }
        return np.array([], dtype=int), new_clenches, info
