import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import numpy as np

class MultiParticipantBoxplotGallery:
    def __init__(self, true_feats_by_participant, false_feats_by_participant, participant_ids=None):
        """
        true_feats_by_participant, false_feats_by_participant:
            lists (length P) where each element is a list/array of length F containing 1-D samples.
            Access like: true_feats_by_participant[p][f]
        participant_ids: optional list of length P with human-readable participant ids (strings)
        """
        self.true = true_feats_by_participant
        self.false = false_feats_by_participant
        assert len(self.true) == len(self.false), "Participant counts must match"
        self.P = len(self.true)
        assert self.P > 0, "Need at least one participant"
        self.F = len(self.true[0])
        # Validate consistent feature counts
        for p in range(self.P):
            assert len(self.true[p]) == self.F and len(self.false[p]) == self.F, "All participants must have same number of features"

        self.participant_ids = participant_ids if participant_ids is not None else [str(i) for i in range(self.P)]
        assert len(self.participant_ids) == self.P

        self.p_idx = 0
        self.f_idx = 0

        # Create figure + axis
        self.fig, self.ax = plt.subplots(figsize=(7, 6))
        plt.subplots_adjust(bottom=0.25)  # room for UI widgets

        # Buttons for feature navigation
        ax_fprev = plt.axes([0.05, 0.12, 0.15, 0.07])
        ax_fnext = plt.axes([0.225, 0.12, 0.15, 0.07])
        self.b_fprev = Button(ax_fprev, 'Prev Feature')
        self.b_fnext = Button(ax_fnext, 'Next Feature')
        self.b_fprev.on_clicked(self.prev_feature)
        self.b_fnext.on_clicked(self.next_feature)

        # Buttons for participant navigation
        ax_pprev = plt.axes([0.45, 0.12, 0.15, 0.07])
        ax_pnext = plt.axes([0.625, 0.12, 0.15, 0.07])
        self.b_pprev = Button(ax_pprev, 'Prev Part')
        self.b_pnext = Button(ax_pnext, 'Next Part')
        self.b_pprev.on_clicked(self.prev_participant)
        self.b_pnext.on_clicked(self.next_participant)

        # TextBox to jump to participant (enter index or id)
        ax_text = plt.axes([0.05, 0.03, 0.4, 0.07])
        self.text_box = TextBox(ax_text, 'Jump to participant (index or id):', initial="")
        self.text_box.on_submit(self.on_text_submit)

        # Keyboard navigation
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Initial draw
        self.draw_current()
        plt.show()

    def draw_current(self):
        self.ax.clear()
        p = self.p_idx
        f = self.f_idx
        data = [self.true[p][f], self.false[p][f]]
        self.ax.boxplot(data, showfliers=False)
        self.ax.set_xticks([1, 2])
        self.ax.set_xticklabels(["True", "False"])
        self.ax.set_ylabel("Value")
        title = f"Feature {f + 1}/{self.F} | Participant {self.participant_ids[p]} ({p+1}/{self.P})"
        self.ax.set_title(title)
        self.ax.grid(axis='y', linestyle=':', linewidth=0.5)
        self.fig.canvas.draw_idle()

    # Feature navigation
    def next_feature(self, event=None):
        self.f_idx = (self.f_idx + 1) % self.F
        self.draw_current()

    def prev_feature(self, event=None):
        self.f_idx = (self.f_idx - 1) % self.F
        self.draw_current()

    # Participant navigation
    def next_participant(self, event=None):
        self.p_idx = (self.p_idx + 1) % self.P
        self.draw_current()

    def prev_participant(self, event=None):
        self.p_idx = (self.p_idx - 1) % self.P
        self.draw_current()

    def on_text_submit(self, text):
        # Try to interpret as index first, otherwise match id
        text = text.strip()
        if text == "":
            return
        # try numeric index (1-based or 0-based)
        try:
            num = int(text)
            # user might type 1-based index
            if 1 <= num <= self.P:
                self.p_idx = num - 1
            elif 0 <= num < self.P:
                self.p_idx = num
            else:
                print(f"Index out of range: {num}")
                return
            self.draw_current()
            return
        except ValueError:
            pass

        # Match by participant id string
        if text in self.participant_ids:
            self.p_idx = self.participant_ids.index(text)
            self.draw_current()
        else:
            print(f"Participant id '{text}' not found. Valid ids: {self.participant_ids}")

    def on_key(self, event):
        # feature nav: left / right / n / p
        if event.key in ('right', 'n'):
            self.next_feature()
        elif event.key in ('left',):
            self.prev_feature()
        # participant nav: up / down / u / d
        elif event.key in ('up',):
            self.next_participant()
        elif event.key in ('down',):
            self.prev_participant()
        # quick exits
        elif event.key == 'escape':
            plt.close(self.fig)

class BoxplotGallery:
    def __init__(self, true_feats, false_feats, participant_id="P001"):
        """
        true_feats, false_feats: lists/arrays of length N where each element is
                                 a 1-D array-like of samples for that feature.
        """
        self.true_feats = true_feats
        self.false_feats = false_feats
        assert len(true_feats) == len(false_feats)
        self.n = len(true_feats)
        self.idx = 0
        self.participant_id = participant_id

        # Create main figure and axis
        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        plt.subplots_adjust(bottom=0.2)  # leave room for buttons

        # Buttons: previous and next
        axprev = plt.axes([0.25, 0.05, 0.2, 0.075])  # left, bottom, width, height
        axnext = plt.axes([0.55, 0.05, 0.2, 0.075])
        self.bprev = Button(axprev, 'Previous')
        self.bnext = Button(axnext, 'Next')

        self.bprev.on_clicked(self.prev)
        self.bnext.on_clicked(self.next)

        # Keyboard navigation
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # draw initial
        self.draw_current()
        plt.show()

    def draw_current(self):
        self.ax.clear()
        i = self.idx
        data = [self.true_feats[i], self.false_feats[i]]
        # Draw the boxplot
        self.ax.boxplot(data, showfliers=False)
        self.ax.set_xticks([1, 2])
        self.ax.set_xticklabels(["True", "False"])
        self.ax.set_ylabel("Value")
        self.ax.set_title(f"Feature {i + 1} Data for {self.participant_id} ({i+1}/{self.n})")
        self.ax.grid(axis='y', linestyle=':', linewidth=0.5)
        self.fig.canvas.draw_idle()

    def next(self, event=None):
        self.idx = (self.idx + 1) % self.n
        self.draw_current()

    def prev(self, event=None):
        self.idx = (self.idx - 1) % self.n
        self.draw_current()

    def on_key(self, event):
        # left/right arrows or n/p
        if event.key in ('right', 'n'):
            self.next()
        elif event.key in ('left', 'p'):
            self.prev()
        elif event.key in ('escape',):
            plt.close(self.fig)

class FeatureAcrossParticipantsGallery(BoxplotGallery):
    """
    Browse a SINGLE feature across MULTIPLE participants.

    true_all / false_all shape:
        true_all[p][f] = 1D array of samples
    """

    def __init__(self, true_all, false_all, feature_idx, participant_ids=None):
        P = len(true_all)
        assert P == len(false_all) and P > 0

        F = len(true_all[0])
        assert 0 <= feature_idx < F, "feature_idx out of range"

        # Build per-participant lists for this feature
        true_feats = [true_all[p][feature_idx] for p in range(P)]
        false_feats = [false_all[p][feature_idx] for p in range(P)]

        self.feature_idx = feature_idx
        self.participant_ids = (
            participant_ids if participant_ids is not None
            else [f"S{i+1}" for i in range(P)]
        )

        # Call parent constructor
        super().__init__(true_feats, false_feats)

    def draw_current(self):
        self.ax.clear()
        i = self.idx
        data = [self.true_feats[i], self.false_feats[i]]
        self.ax.boxplot(data, showfliers=False)
        self.ax.set_xticks([1, 2])
        self.ax.set_xticklabels(["True", "False"])
        self.ax.set_ylabel("Value")
        self.ax.set_title(f"Feature {i + 1} Data for {self.participant_id} ({i + 1}/{self.n})")
        self.ax.grid(axis='y', linestyle=':', linewidth=0.5)

        # <-- fixes -->
        self.ax.set_xlim(0.5, 2.5)  # explicit horizontal room for labels
        self.fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12)  # reduce clipping
        # self.ax.margins(x=0.15)           # alternative to set_xlim

        self.fig.canvas.draw_idle()

class PlotNavigator:
    def __init__(self, subject_list, feature_idx=0):
        self.subjects = list(subject_list.keys())
        self.data = subject_list
        self.feature_idx = feature_idx
        self.idx = 0  # current participant index
        self.fig, self.ax = plt.subplots(figsize=(6,4))
        plt.subplots_adjust(bottom=0.2)
        axprev = plt.axes([0.2, 0.05, 0.2, 0.075])
        axnext = plt.axes([0.6, 0.05, 0.2, 0.075])
        self.bprev = Button(axprev, 'Prev')
        self.bnext = Button(axnext, 'Next')
        self.bprev.on_clicked(self.prev)
        self.bnext.on_clicked(self.next)
        self.update_plot()
        plt.show()

    def update_plot(self):
        participant = self.subjects[self.idx]
        curr_raw = np.asarray(self.data[participant]['X'][:, self.feature_idx])
        y = np.asarray(self.data[participant]['y'].reshape(-1))
        features_true = curr_raw[y == 1]
        features_false = curr_raw[y == 0]
        self.ax.clear()
        self.ax.boxplot([features_true, features_false], showfliers=False)
        self.ax.set_xticks([1,2])
        self.ax.set_xticklabels(["True", "False"])
        self.ax.set_ylabel("Value")
        self.ax.set_title(f"Feature {self.feature_idx+1} Data for {participant}")
        self.fig.canvas.draw_idle()

    def prev(self, event):
        self.idx = (self.idx - 1) % len(self.subjects)
        self.update_plot()

    def next(self, event):
        self.idx = (self.idx + 1) % len(self.subjects)
        self.update_plot()
