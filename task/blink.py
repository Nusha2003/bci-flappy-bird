from psychopy import visual, core, event
import serial
import time

PORT = 'COM8'
BLINK_TRIGGER = 9  
DURATION_MINS = 2
TOTAL_SECONDS = DURATION_MINS * 60

mmbts = serial.Serial(PORT)

win = visual.Window(size=[1200, 800], fullscr=False, monitor="testMonitor", units="deg", color=[-1, -1, -1])

instr = visual.TextStim(win, text="BLINK CALIBRATION\n\nWhen you see 'BLINK', blink naturally.\nKeep your eyes open and still otherwise.\n\nPress any key to start.", height=0.8)
blink_cue = visual.TextStim(win, text="BLINK", color="white", height=3)
rest_cross = visual.TextStim(win, text="+", color="gray", height=2)

instr.draw()
win.flip()
event.waitKeys()

start_time = time.time()

while (time.time() - start_time) < TOTAL_SECONDS:

    rest_cross.draw()
    win.flip()
    core.wait(1.0 + (time.time() % 1.5)) 

    blink_cue.draw()
    win.callOnFlip(mmbts.write, BLINK_TRIGGER)
    win.flip()
    core.wait(0.5)
    
    if 'escape' in event.getKeys():
        break

finish = visual.TextStim(win, text="Blink Calibration Complete!", height=1.5)
finish.draw()
win.flip()
core.wait(3.0)

win.close()
mmbts.close()
core.quit()