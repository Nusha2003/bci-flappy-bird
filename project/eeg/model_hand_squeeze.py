import mne

raw = mne.io.read_raw_edf("/Users/anusha/bci-flappy-bird/data/squeeze_data/allyson_squeeze_raw.edf", preload=False)

print(raw)
print(raw.ch_names)
print(raw.annotations)
print("Number of annotations:", len(raw.annotations))