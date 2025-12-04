import mne, time
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop

raw = mne.io.read_raw_edf("S005R01.edf", preload = True)
data = raw.get_data() * 1e6 #convert Volts -> microvolts
sfreq = int(raw.info["sfreq"])
print(sfreq)
n_channels = data.shape[0]
#(channels, samples)
#64, 10,000
#StreamInfo object stores the declaration of a data stream
info = StreamInfo("FakeEEG", "EEG", n_channels, sfreq, "float32", "")
#print(n_channels)
#print(type)


"""
    creates a fake EEG device
"""
outlet = StreamOutlet(info)

"""
    we have to loop through the file and send chunks. delay = chunk_size/sfreq because -> each sample represents 1/sfreq seconds
    each chunk takes chunk size * 1/sfreq seconds to be samples
    wait that much time before sending the next chunk

"""
chunk_size = 32 
speed = 10
delay = chunk_size/sfreq * speed


idx = 0
while idx < data.shape[1]:
    chunk = data[:,idx:idx+chunk_size].T
    for sample in chunk:
        outlet.push_sample(sample.tolist())
    time.sleep(delay)
    print(f"processing chunk {idx}")
    idx += chunk_size
