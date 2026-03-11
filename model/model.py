import numpy as np
from numpy import load
import re
import os
from collections import Counter
import boxplot
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, make_scorer

class SVMModel:
    def __init__(self, path):
        self.model = None
        self.contents = os.listdir(path)
        self.subject_list = {}
        self.object_info = {}
        self.X_all = []
        self.y_all = []
        self.groups = []

        for content in self.contents:
        # Looping through each folder (of participants)
            dir_path = path + "/" + content
            npz_files = os.listdir(dir_path)

            for file in npz_files:
            # Looping through each .npz files
                data = load(dir_path + "/" + file)
                if not(data["alpha"].shape == (23, 14) and data["labels"].shape == (23, ) and
                      data["channels"].shape == (14, ) and data["sfreq"] == 256):
                    print("Bad data")
                    print(dir_path + "/" + file)

                else:
                    participant_id = re.search(r"S\d{3}", file).group()
                    session_id = re.search(r"E\d{2}", file).group()
                    subject_data = {
                        "subject_id": participant_id,
                        "session_id": session_id,
                        "X": data["alpha"],
                        "y": data["labels"],
                        "channels": data["channels"],
                        "sfreq": data["sfreq"]
                    }

                    self.object_info["feature_num"] = data["alpha"].shape[1]
                    self.subject_list[participant_id] = subject_data

            self.object_info["participant_num"] = len(self.subject_list)
            self.object_info["features_mean"] = np.full((self.object_info["feature_num"], 2, self.object_info["participant_num"]), np.nan, dtype=float)

    def print_shape(self):
        counter = 0
        zero_participants = []
        for participant_id, participant_data in self.subject_list.items():
            X = np.asarray(participant_data["X"])
            y = np.asarray(participant_data["y"]).reshape(-1)
            print(self.object_info)

            print(
                participant_id,
                "X.shape =", X.shape,
                "y.shape =", y.shape,
                "label_counts =", dict(Counter(y.tolist()))
            )

            for feature_idx in range(len(participant_data["X"][0])):
                curr_data = np.asarray(participant_data["X"][:, feature_idx])
                if (curr_data == 0).all():
                    zero_participants.append((participant_id, feature_idx, 1))
                    counter = counter + 1
                elif (curr_data == 0).any():
                    zero_participants.append((participant_id, feature_idx, 0))
                    counter = counter + 1

        print("*** RESULT:", counter, "number of features contained at least one 0")
        for participant_id, feature_idx, all_zero in zero_participants:
            if all_zero:
                print("**", participant_id, "Feature", str(feature_idx + 1), "is all 0")
            else:
                print("**", participant_id, "Feature", str(feature_idx + 1), "contains at least one 0")

    """
        Getting the means of each features for each participants and placing it into a features_mean list (14, 2, 30):
        meaning 14 features, 2 for either 1 or 0 and 30 for 30 participants
        As we loop through the participants, we'll also plot box-whisker plot to see the significance between 1 or 0 (true or false)
        across the 14 features
    """
    def plot_all_whisker(self):
        participants = list(self.subject_list.keys())
        true_all = []
        false_all = []
        participant_ids = []
        for p_idx, participant_id in enumerate(participants):
            pdata = self.subject_list[participant_id]
            X = np.asarray(pdata["X"])
            y = np.asarray(pdata["y"]).reshape(-1)

            # Select trials per class
            true_rows = (y == 1)
            false_rows = (y == 0)

            #Getting the features: splice by columns
            features_true = X[true_rows, :]
            features_false = X[false_rows, :]

            #Inverting the array to bring it to the right dimension (on a row)
            raw_true_features = np.swapaxes(features_true, 0, 1)
            raw_false_features = np.swapaxes(features_false, 0, 1)

            #Add to the array accordingly
            true_all.append(raw_true_features)
            false_all.append(raw_false_features)
            participant_ids.append(participant_id)

        boxplot.MultiParticipantBoxplotGallery(true_all, false_all, participant_ids=participant_ids)

    """
        Plot box-whisker plot for a particular participant:
        outputs all 14 features (comparing true/ false outcomes) for a particular participant (1-30)
    """
    def plot_participant(self, participant_id):
        pdata = self.subject_list[participant_id]
        X = np.asarray(pdata["X"])
        y = np.asarray(pdata["y"]).reshape(-1)

        # Select trials per class
        true_rows = (y == 1)
        false_rows = (y == 0)

        # Getting the features: splice by columns
        features_true = X[true_rows, :]
        features_false = X[false_rows, :]

        # Inverting the array to bring it to the right dimension (on a row)
        raw_true_features = np.swapaxes(features_true, 0, 1)
        raw_false_features = np.swapaxes(features_false, 0, 1)

        boxplot.BoxplotGallery(raw_true_features, raw_false_features, participant_id)

    """
        Plot box-whisker plot for a particular feature:
        outputs all 30 participants performance (comparing true/ false outcomes) for a particular feature (0-13)
    """
    def plot_feature(self, feature_idx):
        raw_true_features = []
        raw_false_features = []
        participant_ids = []

        for participant in self.subject_list.keys():
            curr_raw = np.asarray(self.subject_list[participant]['X'][:, feature_idx])
            y = np.asarray(self.subject_list[participant]['y'].reshape(-1))
            features_true = curr_raw[y == 1]
            features_false = curr_raw[y == 0]

            raw_true_features.append(features_true)
            raw_false_features.append(features_false)
            participant_ids.append(participant)

        boxplot.PlotNavigator(self.subject_list, feature_idx)

    def statistical_analysis(self):
        for p_idx, participant_id in enumerate(list(self.subject_list.keys())):
            pdata = self.subject_list[participant_id]
            X = np.asarray(pdata["X"])
            y = np.asarray(pdata["y"]).reshape(-1)

            # Select trials per class
            true_rows = (y == 1)
            false_rows = (y == 0)

            mean_true = np.mean(X[true_rows, :], axis=0)
            self.object_info["features_mean"][:, 1, p_idx] = mean_true
            mean_false = np.mean(X[false_rows, :], axis=0)
            self.object_info["features_mean"][:, 0, p_idx] = mean_false

        n_features, _, n_participants = self.object_info["features_mean"].shape

        # Prepare arrays for tests:
        p_values = np.full(n_features, np.nan)
        t_stats = np.full(n_features, np.nan)
        cohens_d = np.full(n_features, np.nan)

        for f in range(n_features):
            class0 = self.object_info["features_mean"][f, 0, :]  # shape (n_participants,)
            class1 = self.object_info["features_mean"][f, 1, :]

            # Exclude participants missing either class for this feature
            valid = ~np.isnan(class0) & ~np.isnan(class1)
            n_valid = np.sum(valid)
            if n_valid < 2:
                continue

            v0 = class0[valid]
            v1 = class1[valid]

            # Paired t-test
            t_stat, p_val = stats.ttest_rel(v1, v0)  # compare 1 vs 0; paired
            p_values[f] = p_val
            t_stats[f] = t_stat

            # Cohen's d for paired samples: mean(diff)/std(diff)
            diff = v1 - v0
            cohens_d[f] = np.mean(diff) / np.std(diff, ddof=1)

        # Multiple comparisons correction (FDR)
        reject, pvals_corrected, _, _ = multipletests(np.nan_to_num(p_values, nan=1.0), alpha=0.05, method='fdr_bh')

        # Print summary
        for f in range(n_features):
            print(
                f"Feature {f + 1}: n={np.sum(~np.isnan(self.object_info["features_mean"][f, 0, :]) & ~np.isnan(self.object_info["features_mean"][f, 1, :]))}, "
                f"t={t_stats[f]:.3f}, p={p_values[f]:.4f}, p_fdr={pvals_corrected[f]:.4f}, d={cohens_d[f]:.3f}")

    def prepare_data(self):
        x_list = []
        y_list = []
        group_list = []

        for participant_id, participant_data in self.subject_list.items():
            X = np.asarray(participant_data["X"], dtype=float)
            y = np.asarray(participant_data["y"]).reshape(-1)

            #Normalising dat
            mu = X.mean(axis=0)
            sigma = X.std(axis=0)
            sigma[sigma == 0] = 1.0
            X = (X - mu) / sigma

            x_list.append(X)
            y_list.append(y)
            group_list.append(np.full(len(y), participant_id, dtype=object))

        self.X_all = np.vstack(x_list)
        self.y_all = np.concatenate(y_list)
        self.groups = np.concatenate(group_list)

        print(self.X_all.shape)
        print(self.y_all.shape)
        print(self.groups.shape)

    """
        Plotting the result (runs AFTER Code 1)
        Visualizes the LAST fold's train/test split in PCA-2D space + decision boundary.
    """
    def plot_model(self, X_train, X_test, y_train, y_test):
        # Predictions from the (pipeline) model using the same threshold rule as Code 1
        dec_train = self.model.decision_function(X_train)
        dec_test = self.model.decision_function(X_test)
        yhat_train = (dec_train >= 0).astype(int)
        yhat_test = (dec_test >= 0).astype(int)

        # ----------------------------
        # 1) Reduce to 2D for visualization (PCA) and project BOTH train & test
        # ----------------------------
        pca = PCA(n_components=2, random_state=0)
        X_train_2d = pca.fit_transform(X_train)
        X_test_2d = pca.transform(X_test)  # FIXED: was incorrectly transforming X_train

        # Combine for convenient labeling
        X_vis = np.vstack([X_train_2d, X_test_2d])
        y_vis = np.concatenate([y_train, y_test])
        split = np.array(["train"] * len(X_train_2d) + ["test"] * len(X_test_2d))
        yhat_vis = np.concatenate([yhat_train, yhat_test])

        # ----------------------------
        # 2) Create a 2D grid in PCA space and evaluate the model on it
        #    (map grid points back to original feature space)
        # ----------------------------
        pad = 0.8
        x_min, x_max = X_vis[:, 0].min() - pad, X_vis[:, 0].max() + pad
        y_min, y_max = X_vis[:, 1].min() - pad, X_vis[:, 1].max() + pad

        grid_res = 400  # higher = smoother boundary
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, grid_res),
            np.linspace(y_min, y_max, grid_res)
        )
        grid_2d = np.c_[xx.ravel(), yy.ravel()]

        # Inverse PCA back to original feature space (pipeline will scale internally)
        grid_original = pca.inverse_transform(grid_2d)

        grid_scores = self.model.decision_function(grid_original)
        Z = grid_scores.reshape(xx.shape)

        # ----------------------------
        # 3) Plot: points + decision boundary
        #    - Color = train/test
        #    - Marker shape = actual class (0/1)
        # ----------------------------
        fig, ax = plt.subplots(figsize=(10, 8))

        # Decision boundary at score == 0 (matches your Code 1 threshold)
        ax.contour(xx, yy, Z, levels=[0], linewidths=2)
        # Optional margins
        ax.contour(xx, yy, Z, levels=[-1, 1], linestyles="--", linewidths=1)

        color_map = {"train": "tab:blue", "test": "tab:orange"}
        marker_map = {0: "o", 1: "s"}

        for which_split in ["train", "test"]:
            for cls in [0, 1]:
                mask = (split == which_split) & (y_vis == cls)
                ax.scatter(
                    X_vis[mask, 0],
                    X_vis[mask, 1],
                    s=60,
                    marker=marker_map[cls],
                    c=color_map[which_split],
                    edgecolors="k",
                    linewidths=0.6,
                    alpha=0.95,
                    label=f"{which_split} (y={cls})"
                )

        ax.set_title("SVC Decision Boundary (PCA-2D visualization of LAST fold)")
        ax.set_xlabel("PCA component 1")
        ax.set_ylabel("PCA component 2")
        ax.grid(True, alpha=0.25)

        # Deduplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), loc="best", frameon=True)

        plt.tight_layout()
        plt.show()

    """
        Generating a plot for each participants and comparing the prediction results with the expected values
    """
    def plot_performance(self, y_test, y_pred, participant_id):
        print("y_test:", y_test)
        print("y_pred:", y_pred)
        plt.figure(figsize=(10, 4))
        plt.plot(y_test, label="True Labels", marker="o")
        plt.plot(y_pred, label="Predicted Labels", marker="x")

        plt.title(f"Participant {participant_id} - Predictions")
        plt.xlabel("Sample Index")
        plt.ylabel("Class")
        plt.legend()
        plt.show()

    """
        !!!Before running this function, make sure to run prepare_data() !!!
        The core to the SVM model:
            - Uses LeaveOneGroupOut() strategy
            Pipeline (kernel, C and gamma were determined using each of their performance and picked the most effective model) 
    """
    def run_model(self):
        logo = LeaveOneGroupOut()

        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=1, gamma=1, class_weight="balanced"))
        ])

        ba_scores = []
        cms = []

        for fold, (train_idx, test_idx) in enumerate(logo.split(self.X_all, self.y_all, groups=self.groups)):
            X_train, X_test = self.X_all[train_idx], self.X_all[test_idx]
            y_train, y_test = self.y_all[train_idx], self.y_all[test_idx]
            subject_id = self.groups[test_idx][0]

            # Fit on training subjects
            self.model.fit(X_train, y_train)

            # Decision scores -> threshold at 0 -> predicted labels
            y_score = self.model.decision_function(X_test)  # continuous scores
            y_pred = (y_score >= 0).astype(int)  # threshold = 0

            # Metrics
            ba = balanced_accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

            ba_scores.append(ba)
            cms.append(cm)

            # Printing for debugging purposes:
            pred_dist = dict(zip(*np.unique(y_pred, return_counts=True)))
            print(f"[Fold {fold}] Held-out {subject_id} | BA={ba:.3f} | PRED={pred_dist}")
            print("CM:\n", cm)

            self.plot_model(X_train, X_test, y_train, y_test)
            self.plot_performance(y_test, y_pred, subject_id)

        print("\n------------------------")
        print(f"Mean BA: {np.mean(ba_scores):.3f}")
        print(f"Std BA : {np.std(ba_scores):.3f}")
        print("Aggregate CM:\n", np.sum(cms, axis=0))

model = SVMModel("/Users/lilyh/Documents/USC/Neurotech/alpha_power")
model.prepare_data()
model.run_model()

