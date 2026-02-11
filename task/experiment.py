from psychopy import visual, core, event
import random

# --- Configuration ---
# Doubling the REST and BREAK times to make it "slower"
REST_TIME = 6.0    # Slower baseline
READY_TIME = 2.0   # More time to prepare
FLAP_TIME = 5.0    # Longer window for imagery
BREAK_TIME = 3.0   # Slower recovery

n_trials_per_class = 5
classes = ['Rest', 'Imagine Flap']
trial_list = classes * n_trials_per_class
random.shuffle(trial_list)

# --- Window & Stimuli Setup ---
win = visual.Window(
    size=[1200, 800], 
    fullscr=False, 
    monitor="testMonitor", 
    units="deg", 
    color=[-1, -1, -1] 
)

# Bird Image (Ensure bird.png is in your directory)
bird_img = visual.ImageStim(win, image='/Users/anusha/bci-flappy-bird/task/bird.jpeg', pos=(0, 3), size=(4, 3))

fixation = visual.TextStim(win, text="+", color="white", height=2)
ready_cue = visual.TextStim(win, text="READY", color="yellow", height=2)
task_text = visual.TextStim(win, text="", pos=(0, -2), height=2) # Moved down to fit bird
counter_text = visual.TextStim(win, text="", pos=(10, -7), height=0.7, color="gray")

# --- Instructions ---
intro = visual.TextStim(
    win, 
    text="BCI FLAPPY BIRD CALIBRATION (Slow Mode)\n\n"
         "1. REST: Relax deeply.\n"
         "2. READY: Focus on the screen.\n"
         "3. TASK: Imagine the feeling of flapping.\n\n"
         "Press any key to start.",
    height=0.8
)

intro.draw()
win.flip()
event.waitKeys()

# --- Main Trial Loop ---
for i, trial_type in enumerate(trial_list):
    counter_text.text = f"Trial {i+1}/{len(trial_list)}"
    
    # 1. REST PHASE (Slower)
    timer = core.Clock()
    while timer.getTime() < REST_TIME:
        fixation.draw()
        counter_text.draw()
        win.flip()
    
    # 2. READY PHASE (Slower)
    timer.reset()
    while timer.getTime() < READY_TIME:
        ready_cue.draw()
        counter_text.draw()
        win.flip()
    
    # 3. IMAGINE FLAP PHASE (With Bird Image)
    if trial_type == 'Imagine Flap':
        task_text.text = "FLAP"
        task_text.color = "green"
        show_bird = True
    else:
        task_text.text = "STAY STILL"
        task_text.color = "deepskyblue"
        show_bird = False
        
    timer.reset()
    while timer.getTime() < FLAP_TIME:
        if show_bird:
            bird_img.draw()
        task_text.draw()
        counter_text.draw()
        win.flip()
    
    # 4. BREAK PHASE (Slower)
    win.flip()
    core.wait(BREAK_TIME)
    
    if 'escape' in event.getKeys():
        break

# --- Cleanup ---
finish_text = visual.TextStim(win, text="Calibration Complete!", height=1.5)
finish_text.draw()
win.flip()
core.wait(3.0)

win.close()
core.quit()