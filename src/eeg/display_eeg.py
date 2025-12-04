import matplotlib.pyplot as plt
from collections import deque
from typing import Iterable, Optional
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class EEGWindow:
    fig: plt.Figure
    ax: plt.Axes
    line: any
    buffer_size: int

def init_eeg_window(buffer_size: int, 
                    ylim:tuple[float, float] = (-150, 150), 
                    title: str="EEG Playback (Fp1)",) -> EEGWindow: 
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 4))
    line, = ax.plot([], [], lw=1.5)
    ax.set_xlim(0, buffer_size)
    ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.set_xlabel("Samples (last window)")
    ax.set_ylabel("Amplitude (µV)")
    
    fig.tight_layout()

    return EEGWindow(fig=fig, ax=ax, line=line, buffer_size=buffer_size)

def update_eeg_window(
        window: EEGWindow, 
        signal: np.ndarray,
        blink_markers: Optional[Iterable[int]] = None,) -> None:
    
    signal = np.asarray(signal)

    if signal.shape[0] > window.buffer_size:
        signal = signal[-window.buffer_size:]

    x = np.arange(len(signal))
    window.line.set_xdata(x)
    window.line.set_ydata(signal)

    window.ax.set_xlim(0, max(len(signal), 1))

    if len(window.ax.lines) > 1:
        for extra_line in window.ax.lines[1:]:
            extra_line.remove()
    
    if blink_markers is not None:
        for blink_x in blink_markers:
            if 0 <= blink_x < len(signal):
                window.ax.axvline(blink_x, color='red', linewidth=2)
    
    window.fig.canvas.draw()
    window.fig.canvas.flush_events()