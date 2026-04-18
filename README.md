# BCI Flappy Bird

A brain-computer interface (BCI) controlled Flappy Bird clone built for the USC Neurotech club. The player controls the bird using real EEG signals — either eye blinks or jaw clenches — streamed live from an EEG headset.

---

## Project Scope

The goal is to demonstrate a full, real-time BCI pipeline running end-to-end on consumer EEG hardware: raw signal acquisition → preprocessing → artifact/event detection → game action. No keyboard or mouse input is used during gameplay.

---

## Running the Project

**With EEG hardware (requires LSL stream named `"WS-default"`):**
```bash
conda activate <env-name>
cd project
python main.py
```

**Without hardware (debug mode, Space = inject spike):**
```bash
cd project
python debug/main.py
```

**Environment setup:**
```bash
conda env create -f environment.yml
conda activate <env-name>
```

---

## Project Structure

```
project/
├── main.py                    # Entry point (hardware)
├── high_score.json            # Persisted high score (auto-created)
│
├── eeg/
│   ├── stream.py              # LSL inlet — connects to headset
│   ├── preprocess.py          # Highpass + bandpass filter pipeline
│   ├── blink_detector.py      # Blink detection + calibration
│   └── jaw_clench_detector.py # Jaw clench detection
│
├── controller/
│   └── game_controller.py     # Bridges EEG pipeline → game logic → Qt signals
│
├── game/
│   └── flappy.py              # Pure game logic (no Qt, no EEG)
│
├── ui/
│   ├── test_window.py         # Active UI (MainWindow, all screens)
│   └── main_window.py         # Legacy reference, not loaded
│
└── debug/
    ├── main.py                # Debug entry point
    └── mock_stream.py         # Synthetic EEG stream (no hardware needed)
```

---

## EEG Pipeline

### 1. Signal Acquisition — `eeg/stream.py`

`EEGStream` connects to an LSL outlet named `"WS-default"` at startup. It reads the device's nominal sample rate automatically and exposes a `pull()` method that returns raw multi-channel chunks on demand.

```
Headset → LSL network → EEGStream.pull() → (chunk [N × 8], timestamps [N])
```

The controller polls this every **20 ms** via a Qt timer, accumulating samples into a rolling 5-second buffer.

### 2. Preprocessing — `eeg/preprocess.py`

Each tick, the Fp1 channel (index 1) is extracted from the buffer and passed through a two-stage filter:

| Stage | Type | Cutoff | Purpose |
|---|---|---|---|
| Highpass | Butterworth 4th order | 2 Hz | Remove DC drift and slow baseline wander |
| Bandpass | Butterworth 4th order | 1–10 Hz | Isolate blink-frequency band, reject high-freq noise |

After filtering, the signal is demeaned and polarity-flipped so blink peaks are always positive.

### 3. Blink Detection — `eeg/blink_detector.py`

**Calibration (5 s, blink mode only):**

The user blinks naturally during a 5-second window. After collection:
1. `find_peaks` detects blink candidates using a low initial height threshold
2. Peak heights are trimmed to the **5th–95th percentile** range (discards outliers)
3. The **mean** of the trimmed peaks becomes `calibrated_thresh`

This personalises the threshold to the user's blink amplitude rather than relying on fixed values.

**Detection (per tick):**

`find_peaks` is run on the preprocessed Fp1 signal with:
- Minimum height: `calibrated_thresh` (or adaptive fallback if uncalibrated)
- Peak width: 50–300 ms (rejects noise spikes and slow drifts)
- Refractory period: 500 ms (one jump per blink maximum)
- Merge tolerance: 40 ms (deduplicates peaks seen across overlapping windows)

### 4. Jaw Clench Detection — `eeg/jaw_clench_detector.py`

Jaw clenches produce high-frequency EMG bursts (20–45 Hz) rather than the low-frequency blink artifact.

**Per tick:**
1. Bandpass filter the signal at **20–45 Hz** to isolate EMG
2. Rectify and compute a **smoothed energy envelope** (50 ms RMS window)
3. Compute an adaptive threshold: `max(median + 1.8×MAD, 88th percentile, thresh_min)`
4. `find_peaks` on the envelope with a 350 ms refractory period

**Blink veto:** In jaw mode, a secondary `BlinkDetector` runs in parallel. Any detected blink blocks jaw detections for 250 ms, preventing eye blinks from accidentally triggering jumps.

---

## Game Logic — `game/flappy.py`

The game is fully decoupled from EEG and Qt. It operates on discrete ticks and exposes a minimal API:

```python
game = Game()
game.jump()        # apply upward velocity to bird
game.update()      # advance one tick → returns False on death
game.pipe_data()   # render snapshot of current pipes
game.score         # pipes cleared
```

**Physics constants (tunable in `game/flappy.py`):**

| Constant | Value | Effect |
|---|---|---|
| `GRAVITY` | 0.5 px/tick² | Downward acceleration per tick |
| `JUMP_VEL` | -10.0 px/tick | Upward velocity on flap |
| `PIPE_SPEED` | 2.5 px/tick | Horizontal scroll speed |
| `PIPE_GAP` | 250 px | Vertical opening between pipes |
| `SPAWN_INTERVAL` | 150 ticks | ~2.5 s between pipe spawns at 60 fps |

**Collision:** Bird hitbox is ±14 px horizontal and ±10 px vertical around its centre. Death occurs on pipe collision or hitting the floor. The ceiling clamps the bird without killing it.

---

## Control Modes

| Mode | Signal | Calibration | Jump behaviour |
|---|---|---|---|
| **Blink (1)** | Fp1 low-frequency peak | 5 s required | Single jump per blink |
| **Jaw Clench (2)** | Fp1 EMG envelope burst | None | Hold-to-jump: repeats every 180 ms for 350 ms while clench is active |

---

## Controller — `controller/game_controller.py`

`GameController` is the integration layer between the EEG pipeline and the game. It runs two Qt timers:

- **EEG timer (20 ms):** pull → preprocess → detect → emit `eeg_updated` for the live plot → trigger game action
- **Game timer (16 ms):** advance physics → emit `bird_updated` + `pipes_updated` → detect death → emit `game_over`

During the calibration window, the game timer is paused and raw samples are accumulated for post-calibration processing.

---

## UI — `ui/test_window.py`

Screens are managed by a `QStackedWidget`:

```
HomeMenu  →  CalibrationScreen (blink mode only)  →  PlayScreen  →  GameOverScreen
```

The left panel shows a live Fp1 EEG plot (±150 µV, 2-second rolling window) throughout all screens.

---

## What Has Been Achieved

- Full real-time EEG-to-game pipeline running at ~50–60 Hz on a standard laptop
- Personalised blink threshold calibration using trimmed peak averaging (5th–95th percentile mean)
- Adaptive jaw clench detection with a rolling EMG baseline — no manual threshold tuning needed
- Blink veto in jaw mode to suppress accidental triggers from eye movement
- Persistent high score saved to disk across sessions (`high_score.json`)
- Hardware-free debug mode with synthetic spike injection for UI development and testing
- Modular architecture: game logic, EEG pipeline, and UI are fully decoupled and independently testable
