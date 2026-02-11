import mne

raw = mne.io.read_raw_edf("/Users/anusha/bci-flappy-bird/data/S001R01.edf", preload=True)

# Print all channel names
print(raw.ch_names)

# Pick the indices of Fp1 and Fp2 from the EDF channels
channel_indices = mne.pick_channels(raw.ch_names, include=['Fp1.', 'Fp2.'])
print("Fp1 / Fp2 indices:", channel_indices)
