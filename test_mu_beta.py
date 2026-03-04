import pandas as pd
import numpy as np
from pathlib import Path
import mne
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


path = "/Users/anusha/flappy_bird_data/2.24_avishi_squeeze/2_0001_raw.edf"
raw = mne.io.read_raw_edf(path, preload=True)
ica = mne.preprocessing.ICA(n_components=10, random_state=97)
ica.fit(raw)
ica.exclude = [0] # Usually the first component is the blink
ica.apply(raw)
data = raw.get_data()
#eeg_channels = ['EEG LE-Pz', 'EEG F4-Pz', 'EEG C4-Pz', 'EEG P4-Pz', 'EEG P3-Pz', 'EEG C3-Pz', 'EEG F3-Pz']
#raw.pick_channels(eeg_channels)
raw.filter(l_freq=6.0, h_freq=35.0)
events = mne.find_events(raw, stim_channel='Trigger')
event_id_map = {'Flap': 3, 'Still': 5}
reject = dict(eeg=150e-6)
epochs = mne.Epochs(raw, events, event_id=event_id_map, 
                    tmin=-1.0, tmax=8.0, baseline=None, preload=True)
epochs.pick(['EEG C3-Pz', 'EEG C4-Pz'])


bands = {
    'Mu (8-11 Hz)': (8, 11),
    'Beta (14-18 Hz)': (14, 18)
}

freqs = np.arange(7, 30)  
n_cycles = freqs / 2.

tfr_flap = mne.time_frequency.tfr_morlet(
    epochs['Flap'], freqs=freqs, n_cycles=n_cycles, 
    return_itc=False, average=True
)

tfr_still = mne.time_frequency.tfr_morlet(
    epochs['Still'], freqs=freqs, n_cycles=n_cycles, 
    return_itc=False, average=True
)

tfr_flap.apply_baseline(baseline=(None, 0), mode='percent')
tfr_still.apply_baseline(baseline=(None, 0), mode='percent')


fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

for i, (name, (fmin, fmax)) in enumerate(bands.items()):
    # 1. Process Flap Data
    tfr_band_flap = tfr_flap.copy().crop(fmin=fmin, fmax=fmax)
    ch_idx = epochs.ch_names.index('EEG C3-Pz')
    erd_flap = tfr_band_flap.data[ch_idx].mean(axis=0) * 100 
    
    # 2. Process Still Data
    tfr_band_still = tfr_still.copy().crop(fmin=fmin, fmax=fmax)
    erd_still = tfr_band_still.data[ch_idx].mean(axis=0) * 100 
    erd_flap_smooth = gaussian_filter1d(erd_flap, sigma=7)
    erd_still_smooth = gaussian_filter1d(erd_still, sigma=7)
    
    # 4. Plot both conditions
    ax[i].plot(tfr_flap.times, erd_flap_smooth, label=f'Flap {name}', color='red', linewidth=2)
    ax[i].plot(tfr_still.times, erd_still_smooth, label=f'Still {name}', color='blue', linewidth=2)
    
    # Formatting
    ax[i].axhline(0, color='black', linestyle='--')
    ax[i].axvline(0, color='gray', linestyle=':', label='Instruction Start')
    ax[i].set_ylabel('ERD %')
    ax[i].set_title(f'ERD/ERS - C3 ({name})')
    ax[i].grid(True, alpha=0.3)
    ax[i].legend(loc='upper right')

ax[1].set_xlabel('Time (Seconds)')
plt.tight_layout()