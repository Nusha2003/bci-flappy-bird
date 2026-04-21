import mne

edf = "/Users/anusha/bci-flappy-bird/data/mendeley data/S010/S010E01.edf"

raw = mne.io.read_raw_edf(edf, preload=True)

print("Channels:", raw.ch_names)
print("Channel types:", raw.get_channel_types())
print("Sampling rate:", raw.info["sfreq"])
print("Data shape:", raw.get_data().shape)