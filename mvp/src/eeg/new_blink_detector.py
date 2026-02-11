import time
import uuid

from matplotlib import pyplot as plt
from mne import set_log_level

from mne_lsl.datasets import sample
from mne_lsl.player import PlayerLSL as Player
from mne_lsl.stream import StreamLSL as Stream


stream = Stream(bufsize=2, name = "EEG").connect()
stream.info

