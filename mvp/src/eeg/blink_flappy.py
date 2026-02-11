import time
import threading
import numpy as np
from collections import deque
from scipy.signal import butter, filtfilt, find_peaks
import pylsl

##############################
# PYGAME IMPORTS
##############################
import pygame
from sys import exit


############################################################
# 1. BLINK DETECTION THREAD
############################################################

def highpass(x, fs):
    b, a = butter(4, 2 / (fs / 2), btype='high')
    return filtfilt(b, a, x)

def bandpass(x, fs):
    b, a = butter(4, [1 / (fs / 2), 10 / (fs / 2)], btype='band')
    return filtfilt(b, a, x)


# GLOBAL FLAG shared with game thread
blink_detected = False
last_blink_time = 0
refractory = 0.35   # no double-blinks


def blink_detector():
    global blink_detected, last_blink_time

    print("Looking for FakeEEG stream...")
    streams = pylsl.resolve_byprop("name", "FakeEEG", timeout=10)
    if not streams:
        print("No FakeEEG stream found.")
        return

    inlet = pylsl.StreamInlet(
        streams[0],
        max_buflen=5,
        processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter
    )

    fs = int(inlet.info().nominal_srate())
    fp1 = 21
    buffer = deque(maxlen=fs * 5)

    while True:
        chunk, ts = inlet.pull_chunk(max_samples=int(fs / 3), timeout=0.0)
        if not ts:
            continue

        chunk = np.array(chunk)
        fp1_signal = chunk[:, fp1]

        for v in fp1_signal:
            buffer.append(v)

        if len(buffer) < fs:
            continue

        ########################################
        # PROCESS EEG WINDOW
        ########################################
        data = np.array(buffer)[-fs*5:] * 1e6

        hp = highpass(data, fs)
        bp = bandpass(hp, fs)
        signal = bp - np.mean(bp)

        # Correct sign of blink
        if np.max(signal) < abs(np.min(signal)):
            signal = -signal

        # Absolute threshold to prevent small peaks being detected
        dynamic_thresh = (np.max(signal) - np.min(signal)) * 0.25
        thresh = max(dynamic_thresh, 80)     # ensure minimum amplitude

        peaks, props = find_peaks(
            signal,
            height=thresh,
            distance=int(0.25 * fs)  # blinks can't occur too close
        )

        if len(peaks) > 0:
            now = time.time()
            if now - last_blink_time > refractory:
                blink_detected = True
                last_blink_time = now


############################################################
# 2. SIMPLE BLINK-CONTROLLED FLAPPY BIRD DEMO
############################################################

GAME_WIDTH = 360
GAME_HEIGHT = 640
bird_width = 34
bird_height = 24


def run_game():

    pygame.init()
    window = pygame.display.set_mode((GAME_WIDTH, GAME_HEIGHT))
    pygame.display.set_caption("Blink → Jump Demo")
    clock = pygame.time.Clock()

    # Load assets
    bg = pygame.image.load("/Users/anusha/bci-flappy-bird/src/game/flappybirdbg.png")
    bird_img = pygame.transform.scale(
        pygame.image.load("/Users/anusha/bci-flappy-bird/src/game/flappybird.png"),
        (bird_width, bird_height)
    )

    # Bird position
    bird_x = GAME_WIDTH / 8
    bird_y = GAME_HEIGHT / 2
    global blink_detected

    jump_strength = 15     # small jump
    gravity = 1.0          # mild gravity
    vel_y = 0

    while True:

        # Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        ################################
        # BLINK → JUMP
        ################################
        if blink_detected:
            vel_y = -jump_strength
            blink_detected = False

        ################################
        # PHYSICS
        ################################
        vel_y += gravity
        bird_y += vel_y

        # keep inside screen
        bird_y = max(0, min(GAME_HEIGHT - bird_height, bird_y))

        ################################
        # DRAWING
        ################################
        window.blit(bg, (0, 0))
        window.blit(bird_img, (bird_x, bird_y))

        font = pygame.font.SysFont("Arial", 28)
        window.blit(font.render("Blink to make bird jump!", True, (255, 255, 255)),
                    (20, 20))

        pygame.display.update()
        clock.tick(60)


############################################################
# 3. START THREAD + GAME
############################################################

if __name__ == "__main__":
    t = threading.Thread(target=blink_detector, daemon=True)
    t.start()
    run_game()
