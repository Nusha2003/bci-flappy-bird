from psychopy import visual, core, event
import random
import serial


PORT = 'COM6'

REST_TRIGGER = 1
READY_TRIGGER = 2
SQUEEZE_TRIGGER = 3
BREAK_TRIGGER = 4
STILL_TRIGGER = 5

#manages serial port connections
mmbts = serial.Serial()
mmbts.port = PORT
mmbts.open()




REST_TIME = 2.0    
READY_TIME = 2.0   
TASK_TIME = 4.0    
BREAK_TIME = 2.0   


n_trials_per_class = 30
classes = ['Rest', 'Squeeze']
trial_list = classes * n_trials_per_class
random.shuffle(trial_list)

#represents a window for displaying one or more stimuli
win = visual.Window(
    size=[1200, 800], 
    fullscr=False, 
    monitor="testMonitor", 
    units="deg", 
    color=[-1, -1, -1] 
)
"""
def send_trigger(code):
    mmbts.write(bytes([code]))
    mmbts.flush()

    """
#text stimuli to be displayed in a window
#bird_img = visual.ImageStim(win, image='/Users/anusha/bci-flappy-bird/task/bird.jpeg', pos=(0, 3), size=(4, 3))
fixation = visual.TextStim(win, text="+", color="white", height=2)
ready_cue = visual.TextStim(win, text="READY", color="yellow", height=2)
task_text = visual.TextStim(win, text="", pos=(0, -2), height=2) # Moved down to fit bird
counter_text = visual.TextStim(win, text="", pos=(10, -7), height=0.7, color="gray")


intro = visual.TextStim(
    win, 
    text="BCI FLAPPY BIRD CALIBRATION (Slow Mode)\n\n"
         "1. REST: Relax deeply.\n"
         "2. READY: Focus on the screen.\n"
         "3. TASK: Squeeze your hand tightly.\n\n"
         "Press any key to start.",
    height=0.8
)

intro.draw()
#nothing appears on the screen until you call flip()
#use callonFlip
win.flip()
event.waitKeys()
timer = core.Clock()

fixation.draw()
win.callOnFlip(mmbts.write, REST_TRIGGER)
win.flip()
core.wait(5.0)


#reset before the trials start?

for i, trial_type in enumerate(trial_list):
    counter_text.text = f"Trial {i+1}/{len(trial_list)}"
    
    #rest phase

    fixation.draw()
    win.callOnFlip(mmbts.write, bytes([REST_TRIGGER]))
    win.flip()

    core.wait(REST_TIME)

    ready_cue.draw()
    counter_text.draw()
    win.callOnFlip(mmbts.write, bytes([READY_TRIGGER]))
    win.flip()
    core.wait(READY_TIME)

    #imagine flap
    if trial_type == 'Squeeze':
        task_text.text = "Squeeze"
        task_text.color = "green"
        #show_bird = True
        cue_trigger = SQUEEZE_TRIGGER
    else:
        #motor inhibition
        task_text.text = "Stay Still"
        task_text.color = "deepskyblue"
        #show_bird = False
        cue_trigger = STILL_TRIGGER
      
    """if show_bird:
            bird_img.draw()"""

    task_text.draw()
    counter_text.draw()
    win.callOnFlip(mmbts.write, bytes([cue_trigger]))
    win.flip()
    core.wait(TASK_TIME)


    win.callOnFlip(mmbts.write, bytes([BREAK_TRIGGER]))
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
mmbts.close()