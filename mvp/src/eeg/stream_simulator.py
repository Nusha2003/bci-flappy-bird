import mne, time
from pylsl import StreamInfo, StreamOutlet

raw = mne.io.read_raw_edf("/Users/anusha/bci-flappy-bird/data/S001R01.edf", preload=True)
data = raw.get_data()
sfreq = int(raw.info["sfreq"])
n_channels = data.shape[0]

info = StreamInfo("FakeEEG", "EEG", n_channels, sfreq, "float32", "")
outlet = StreamOutlet(info)

idx = 0
total_samples = data.shape[1]   # this is already 1 minute of data

while idx < total_samples:       # <--- stop when finished
    sample = data[:, idx]
    outlet.push_sample(sample.tolist())
    idx += 1
    time.sleep(1 / sfreq)

print("Finished streaming entire 60-second EEG file.")
