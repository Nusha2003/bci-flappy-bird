import os
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import mne
import numpy as np


class EEGData:
    def __init__(self, l_freq=8.0, h_freq=30.0):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.event_id_map = {"Squeeze": 3, "Still": 5}
        self.eeg_channels = [
            "EEG F4-Pz",
            "EEG C4-Pz",
            "EEG P4-Pz",
            "EEG P3-Pz",
            "EEG C3-Pz",
            "EEG F3-Pz",
        ]

    def load_epochs(self, file_path, tmin=0.0, tmax=1.0):
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        raw.filter(l_freq=self.l_freq, h_freq=self.h_freq, verbose=False)

        if "Trigger" not in raw.ch_names:
            raise ValueError("Expected a Trigger channel in squeeze_data EDF format.")

        events = mne.find_events(raw, stim_channel="Trigger", verbose=False)
        if len(events) == 0:
            raise ValueError("No trigger events were found in this EDF.")

        event_codes = set(events[:, 2].tolist())
        expected_codes = {self.event_id_map["Squeeze"], self.event_id_map["Still"]}
        if not expected_codes.issubset(event_codes):
            raise ValueError(
                f"Missing squeeze/still trigger codes. Found trigger codes: {sorted(event_codes)}"
            )

        if not all(channel in raw.ch_names for channel in self.eeg_channels):
            raise ValueError(
                "EDF does not match the expected squeeze_data channel layout. "
                f"Available channels: {raw.ch_names}"
            )

        raw.pick(self.eeg_channels)
        epochs = mne.Epochs(
            raw,
            events,
            event_id=self.event_id_map,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            preload=True,
            verbose=False,
        )
        return epochs

    def load_epoch_data(self, file_path, tmin=0.0, tmax=1.0):
        epochs = self.load_epochs(file_path, tmin=tmin, tmax=tmax)
        X = epochs.get_data(copy=True)
        y = self._encode_labels(epochs.events[:, -1])
        return X, y

    def _encode_labels(self, event_codes):
        event_codes = np.asarray(event_codes)
        return np.where(event_codes == self.event_id_map["Squeeze"], 1, 0)


def resolve_edf_files(path: Path):
    if path.is_dir():
        files = sorted(path.glob("*.edf"))
    else:
        files = [path]

    if not files:
        raise ValueError(f"No EDF files found at {path}")
    return files
