import numpy as np
import mne
from pathlib import Path

class EEGData:
    def __init__(self, l_freq=6.0, h_freq=35.0):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.event_id_map = {'Flap': 3, 'Still': 5}
        self.eeg_channels = ['EEG C4-Pz', 'EEG C3-Pz']

    def load_and_process(self, file_path, tmin=-1.0, tmax=4.0):
        """reads edf, filters, and returns features and labels."""

        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        raw.filter(l_freq=self.l_freq, h_freq=self.h_freq, verbose=False)
        events = mne.find_events(raw, stim_channel='Trigger', verbose=False)
        raw.pick_channels(self.eeg_channels)
        epochs = mne.Epochs(raw, events, event_id=self.event_id_map, 
                            tmin=tmin, tmax=tmax, baseline=None, 
                            preload=True, verbose=False)
        X = self._extract_features(epochs)
        y = epochs.events[:, -1]
        
        return X, y

    def _extract_features(self, epochs):
        """internal helper to extract mu and Beta log-variance."""
        # Extract Mu (8-13 Hz)
        mu_data = epochs.copy().filter(8.0, 13.0, verbose=False).get_data()
        X_mu = np.log(np.var(mu_data, axis=2)) 

        # Extract Beta (15-30 Hz)
        beta_data = epochs.copy().filter(15.0, 30.0, verbose=False).get_data()
        X_beta = np.log(np.var(beta_data, axis=2)) 

        # Combine into [C4_Mu, C3_Mu, C4_Beta, C3_Beta]
        return np.hstack((X_mu, X_beta))