import os
import mne
import numpy as np
from scipy.signal import welch



def compute_alpha_power(segment, sfreq):
    # Ensure segment is 2D (Channels, Time)
    if segment.ndim == 1:
        segment = segment.reshape(1, -1)
    
    # Welch returns psd as (n_channels, n_freqs)
    freqs, psd = welch(
        segment,
        fs=sfreq,
        nperseg=int(sfreq * 2),
        axis=-1 # Explicitly use the last axis
    )

    # alpha_mask is 1D (length = n_freqs)
    alpha_mask = (freqs >= 8) & (freqs <= 12)

    # Indexing: psd[:, alpha_mask] selects all channels, but only alpha frequencies
    # Result of psd[:, alpha_mask] is (n_channels, n_alpha_freqs)
    alpha_values = psd[:, alpha_mask]
    
    # Mean across the frequency dimension (axis 1)
    alpha = np.mean(alpha_values, axis=1)

    return alpha




def process_edf(edf_path, output_path,
                window_size=10,
                overlap=0.5):

    print(f"Processing: {edf_path}")

    # Load
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    sfreq = raw.info["sfreq"]
    channels = raw.ch_names

    # Keep first 2 min
    raw.crop(0, 120)

    # Filters
    raw.filter(1, 40, verbose=False)
    raw.notch_filter(60, verbose=False)

    data = raw.get_data()

    n_channels, n_samples = data.shape

    # Window params
    win = int(window_size * sfreq)
    step = int(win * (1 - overlap))

    trials = []
    alpha_feats = []
    labels = []

    # Sliding windows
    for start in range(0, n_samples - win + 1, step):

        end = start + win

        seg = data[:, start:end]

        # Baseline normalize
        mean = np.mean(seg, axis=1, keepdims=True) + 1e-8
        seg = seg / mean
        assert seg.ndim == 2, f"Segment shape wrong: {seg.shape}"
        # Alpha power
        alpha = compute_alpha_power(seg, sfreq)

        # Label
        mid_time = (start + end) / 2 / sfreq

        if mid_time < 60:
            label = 0   # relax
        else:
            label = 1   # task

        trials.append(seg)
        alpha_feats.append(alpha)
        labels.append(label)

    # Convert
    trials = np.array(trials)
    alpha_feats = np.array(alpha_feats)
    labels = np.array(labels)

    # Save
    np.savez_compressed(
        output_path,
        trials=trials,
        alpha=alpha_feats,
        labels=labels,
        sfreq=sfreq,
        channels=channels
    )

    print(f"Saved -> {output_path}")
    print(f"Trials: {trials.shape}\n")



def process_dataset(
    root_dir,
    output_root="processed"
):

    os.makedirs(output_root, exist_ok=True)

    for subject in sorted(os.listdir(root_dir)):

        subj_path = os.path.join(root_dir, subject)

        if not os.path.isdir(subj_path):
            continue

        print(f"\n=== Subject {subject} ===")

        out_subj = os.path.join(output_root, subject)
        os.makedirs(out_subj, exist_ok=True)

        for file in sorted(os.listdir(subj_path)):

            if not file.endswith(".edf"):
                continue

            edf_path = os.path.join(subj_path, file)

            base = file.replace(".edf", ".npz")
            out_file = os.path.join(out_subj, base)

            try:
                process_edf(edf_path, out_file)

            except Exception as e:
                print(f"❌ Failed: {file}")
                print(e)

if __name__ == "__main__":

    DATA_ROOT = "/Users/anusha/bci-flappy-bird/data/mendeley data"
    OUTPUT_ROOT = "/Users/anusha/bci-flappy-bird/data/alpha_power"

    process_dataset(DATA_ROOT, OUTPUT_ROOT)
