from psychopy import visual, core, event
import serial
import time

PORT = 'COM3'
CLENCH_TRIGGER = 10  
DURATION_MINS = 2
TOTAL_SECONDS = DURATION_MINS * 60

mmbts = serial.Serial(PORT)

win = visual.Window(size=[1200, 800], fullscr=False, monitor="testMonitor", units="deg", color=[-1, -1, -1])

instr = visual.TextStim(
    win,
    text="JAW CLENCH CALIBRATION\n\nWhen you see 'CLENCH', gently clench your jaw.\nRelax your jaw otherwise.\n\nPress any key to start.",
    height=0.8
)

clench_cue = visual.TextStim(win, text="CLENCH", color="white", height=3)
rest_cross = visual.TextStim(win, text="+", color="gray", height=2)

instr.draw()
win.flip()
event.waitKeys()

start_time = time.time()

while (time.time() - start_time) < TOTAL_SECONDS:

    rest_cross.draw()
    win.flip()
    core.wait(1.0 + (time.time() % 1.5)) 

    clench_cue.draw()
    win.callOnFlip(mmbts.write, bytes([CLENCH_TRIGGER]))
    win.flip()
    core.wait(0.5)
    
    if 'escape' in event.getKeys():
        break

finish = visual.TextStim(win, text="Jaw Clench Calibration Complete!", height=1.5)
finish.draw()
win.flip()
core.wait(3.0)

win.close()
mmbts.close()
core.quit()