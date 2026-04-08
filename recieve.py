import os
import time

from datetime import datetime

import pandas as pd
import pylsl
import numpy as np


DEFAULT_DURATION = 50
DEFAULT_STREAM_NAME = 'WS-default'
DEFAULT_OUTPUT_PATH='C:\\Users\\winni\\Desktop\\Neurotech\\bci-flappy-bird\\mvp'

def find_stream(stream_name: str) -> pylsl.StreamInlet:
    """
    Finds the LSL stream by name and returns a StreamInlet for data collection.

    Args:
        stream_name (str): The name of the stream to find.

    Returns:
        dsi_stream_inlet (pylsl.StreamInlet): The inlet for the found stream.
    """

    print("Looking for EEG streams")
    streams = pylsl.resolve_byprop(prop="name", value=stream_name, timeout=10)
    # If timeout, end stream to prevent terminal from being frozen.
    if len(streams) == 0:
        raise Exception(
            f"Could not find stream name {stream_name}. Ending now..."
        )

    dsi_stream = None
    num_streams = len(streams)
    print(f"Found {num_streams} stream(s):")

    if num_streams > 1:
        raise Exception(
            f"{num_streams} found. Expected one Stream. Please close "
            f"other streams."
        )

    for _, stream in enumerate(streams):
        print(f"Name: '{stream.name()}'")
        dsi_stream = stream

    if not dsi_stream:
        raise Exception("No DSI stream found.")

    dsi_stream_inlet = pylsl.StreamInlet(dsi_stream)
    return dsi_stream_inlet

"""
def receive_data(
    stream: pylsl.StreamInlet, output_path: str, duration: float
) -> None:
    Python script to record data from Wearable Sensing LSL stream (dsi2lsl).
    Records for specified duration and saves CSV to desired path.

    Args:
        stream (pylsl.StreamInlet): The LSL stream inlet to read data from.
        output_path (str): The path where the CSV file will be saved.
        duration (float): The duration in seconds for which to collect data.

    Returns:
        None
    
    try:
        # Get stream metadata.
        info = stream.info()
        print(f"Stream info: {info.name()} ({info.type()})")
        # Print stream metadata from the desc xml
        print(f"Stream description: {info.as_xml()}")

        # Generate unique filename for new CSV file.
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        unique_filename = f"DSIdata-{duration}s-{timestamp}-{info.name()}.csv"
        full_path = os.path.join(output_path, unique_filename)

        # Get channel labels.
        ch = info.desc().child("channels").child("channel")
        units = ch.child_value("unit")
        labels = []
        for _ in range(info.channel_count()):
            labels.append(ch.child_value("label"))
            ch = ch.next_sibling()

        # Create column names in same row as channels.
        columns = ["Timestamp"] + labels + ["lsl_timestamp"]

        # Collect all the data first.
        all_data = []
        start_time = time.time()
        sample_counter = 1

        print(
            f"\nCollecting data for {duration}s... "
            "(Interrupt [Ctrl-C] to stop)\n"
        )
        # Loop records data for duration, ensures each row is paired.
        while time.time() - start_time < duration:
            samples, timestamps = stream.pull_chunk()
            print(np.array(samples).shape)
            if samples:
                for sample, lsl_timestamp in zip(samples, timestamps):
                    row = [sample_counter] + sample + [lsl_timestamp]
                    all_data.append(row)
                    sample_counter += 1

        # Create DataFrame and save to CSV.
        df = pd.DataFrame(all_data, columns=columns)

        reference_label = info.desc().child("reference").child_value("label")
        # Write the metadata to the CSV file header
        with open(full_path, "w", newline="") as f:
            f.write(f"stream_name,{info.name()}\n")
            f.write(f"daq_type,{info.type()}\n")
            f.write(f"units,{units}\n")

            f.write(f"reference,{reference_label}\n")
            f.write(f"sample_rate,{info.nominal_srate()}\n")
            df.to_csv(f, index=False)

        print("\nRecording finished.")
        print(f"Saved {len(df)} samples to {full_path}")

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C). Exiting gracefully...")
    except Exception as e:
        print(f"Error: {e}")
"""

if __name__ == "__main__":
    """Main entry point for the script to collect data from a DSI stream.
    Parses command line arguments for output path, stream name, and duration.
    If no arguments are provided, uses default values.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="A script that collects data from a DSI stream and writes "
        "it a file."
    )
    parser.add_argument(
        "--output",
        type=str,
        help="The path where data should be written to.",
        default=DEFAULT_OUTPUT_PATH,
    )
    parser.add_argument(
        "--stream",
        type=str,
        help="The stream name configured in the LSL app.",
        default=DEFAULT_STREAM_NAME,
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_DURATION,
        help="The duration in seconds for the data collection to run "
        "(default: 30).",
    )
    args = parser.parse_args()

    stream_info = find_stream(args.stream)
    while True:
        samples, timestamps = stream_info.pull_chunk(timeout=1.0)
        if samples:
            data=np.array(samples)
            print("Chunk received:", data.shape)