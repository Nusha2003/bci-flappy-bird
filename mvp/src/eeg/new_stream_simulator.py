import time
import uuid

from matplotlib import pyplot as plt
from mne import set_log_level


from mne_lsl.datasets import sample
from mne_lsl.player import PlayerLSL as Player


set_log_level("WARNING")
source_id = uuid.uuid4().hex
fname = "/Users/anusha/flappy_bird_data/2.23_allyson_squeeze/2_0002_raw.edf"
player = Player(fname, chunk_size=32, name = "EEG").start()
print("Player info:", player.info)
sfreq = player.info["sfreq"]
chunk_size = player.chunk_size
interval = chunk_size / sfreq  # in seconds
print(f"Interval between 2 push operations: {interval} seconds.")

