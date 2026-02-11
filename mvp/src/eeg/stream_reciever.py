from pylsl import StreamInlet, resolve_byprop
import numpy as np

print("Looking for FakeEEG stream...")
streams = resolve_byprop("name", "FakeEEG")
print("Found:", streams)

inlet = StreamInlet(streams[0])
print("Connected. Listening...")

while True:
    samples, timestamps = inlet.pull_chunk(timeout=1.0)
    if samples:
        data = np.array(samples)
        print("Chunk received:", data.shape)