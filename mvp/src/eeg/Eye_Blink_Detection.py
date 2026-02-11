import mne
import numpy as np
import matplotlib.pyplot as plt

#I think this will need to change when we push the code
file_path = "/Users/anusha/bci-flappy-bird/data/S001R01.edf"


def detect_blinks(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True)
    raw.filter(0.5, 8.0, fir_design="firwin")
    # AUTO-DETECT FRONTAL CHANNELS
    prefixes = ("Fp", "AF", "F") 
    all_channels = raw.ch_names
    frontal_channels = [
        ch for ch in all_channels 
        if ch.upper().startswith(prefixes)]
    
    #Ensure frontal channels are present and output if none are detectd
    available = [ch for ch in frontal_channels if ch in raw.ch_names]
    if len(available) == 0:
        raise ValueError("Cannot detect blinks. No frontal channels found in")
    raw_frontal = raw.copy().pick_channels(available)
    data = raw_frontal.get_data()

   #Identify peaks/blinks in the frontal channels 
    frontal_signal = np.mean(data, axis=0)  # average frontal activity
    threshold = 2 * np.std(frontal_signal) # Use threshold = N * standard deviation
    blink_indices = np.where(frontal_signal > threshold)[0]# detect blinks
    blink_times = raw.times[blink_indices] # convert sample indices to times
    
    # This accounts for blinks that are close together, and it consolidates the 260ms of blinking into one bool that indicates a blink
    min_blink_interval = 0.26  # this defines the minimum time between blinks 
    filtered_blinks = []
    last_blink_time = -np.inf
    for t in blink_times:
        if t - last_blink_time >= min_blink_interval:
            filtered_blinks.append(t)
            last_blink_time = t
    
    # THIS OUTPUTS A BOOL and FLOAT, indicating when (what time, t) and IF (bool) Flappy should jump
    blink_events = []
    for t in filtered_blinks:
        blink_events.append({
            "jump": True,  # THis indicates to tap/jump
            "time": float(t)
        })

    # CHAT gave me this to plot the results, but not necessary visualization for the game
    plt.figure(figsize=(12, 6))
    plt.plot(raw.times, frontal_signal, label="Frontal EEG (avg)")
    plt.scatter(filtered_blinks,
            [frontal_signal[np.where(raw.times == t)[0][0]] for t in filtered_blinks],
            color='red', label="Detected blinks")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.title("Eye Blink Detection")
    plt.legend()
    plt.show()
    return blink_events, frontal_signal, raw

#this prints the detected blinks so we can see them and makes sure they're correct. We can change this to ouptu them into the game
print("Detected blink events:")
blink_events, frontal_signal, raw = detect_blinks(file_path)
for event in blink_events:
    print(event)
