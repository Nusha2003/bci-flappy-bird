import argparse
import os
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import joblib
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

try:
    from model.preprocess import EEGData, resolve_edf_files
except ImportError:
    from preprocess import EEGData, resolve_edf_files


DEFAULT_DATA_DIR = Path("/Users/anusha/bci-flappy-bird/data/squeeze_data")
DEFAULT_MODEL_PATH = Path(
    "/Users/anusha/bci-flappy-bird/project/eeg/hand_clench_csp_lda.joblib"
)


def build_pipeline(n_components=4):
    return Pipeline(
        [
            ("csp", CSP(n_components=n_components, log=True, norm_trace=False)),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )


def load_dataset(proc: EEGData, files, tmin=0.0, tmax=1.0):
    X_parts = []
    y_parts = []
    used_files = []
    skipped_files = []

    for edf_file in files:
        try:
            X_file, y_file = proc.load_epoch_data(edf_file, tmin=tmin, tmax=tmax)
        except ValueError as exc:
            skipped_files.append((edf_file, str(exc)))
            print(f"Skipped {edf_file.name}: {exc}")
            continue

        print(f"Loaded {edf_file.name}: X={X_file.shape}, y={y_file.shape}")
        X_parts.append(X_file)
        y_parts.append(y_file)
        used_files.append(edf_file)

    if not X_parts:
        raise ValueError("No compatible EDF files were found in the provided path.")

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    return X, y, used_files, skipped_files


def train_model(
    data_path: Path = DEFAULT_DATA_DIR,
    output_model_path: Path = DEFAULT_MODEL_PATH,
    csp_components: int = 4,
    tmin: float = 0.0,
    tmax: float = 1.0,
    random_state: int = 42,
):
    proc = EEGData()
    edf_files = resolve_edf_files(data_path)
    X, y, used_files, skipped_files = load_dataset(proc, edf_files, tmin=tmin, tmax=tmax)

    pipeline = build_pipeline(n_components=csp_components)
    pipeline.fit(X, y)

    artifact = {
        "pipeline": pipeline,
        "channels": proc.eeg_channels,
        "sfreq": None,
        "tmin": tmin,
        "tmax": tmax,
        "window_samples": int(X.shape[-1]),
        "csp_components": csp_components,
        "used_files": [str(path) for path in used_files],
        "skipped_files": [(str(path), reason) for path, reason in skipped_files],
        "event_id_map": proc.event_id_map,
    }

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_model_path)
    return artifact, X, y, used_files, skipped_files


def main():
    parser = argparse.ArgumentParser(
        description="Train and save a squeeze-vs-still CSP + LDA model."
    )
    parser.add_argument(
        "data_path",
        nargs="?",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to one EDF file or to the squeeze_data directory",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Where to save the trained model artifact",
    )
    parser.add_argument(
        "--csp-components",
        type=int,
        default=4,
        help="Number of CSP components to keep",
    )
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=1.0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    proc = EEGData()
    edf_files = resolve_edf_files(args.data_path)
    X, y, used_files, skipped_files = load_dataset(
        proc, edf_files, tmin=args.tmin, tmax=args.tmax
    )

    print(f"Used {len(used_files)} EDF file(s) from: {args.data_path}")
    if skipped_files:
        print(f"Skipped {len(skipped_files)} EDF file(s) with incompatible data.")
    print(f"Epoch tensor shape: {X.shape}")
    print(f"Label shape: {y.shape}")
    print(
        "Class counts:",
        {"Still (0)": int(np.sum(y == 0)), "Squeeze (1)": int(np.sum(y == 1))},
    )

    pipeline = build_pipeline(n_components=args.csp_components)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    print(f"CV accuracy: {cv_scores.mean():.2%} +/- {cv_scores.std():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )
    pipeline.fit(X_train, y_train)
    test_accuracy = pipeline.score(X_test, y_test)
    y_pred = pipeline.predict(X_test)
    print(f"Held-out accuracy: {test_accuracy:.2%}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=["Still", "Squeeze"]))

    artifact = {
        "pipeline": pipeline,
        "channels": proc.eeg_channels,
        "sfreq": None,
        "tmin": args.tmin,
        "tmax": args.tmax,
        "window_samples": int(X.shape[-1]),
        "csp_components": args.csp_components,
        "used_files": [str(path) for path in used_files],
        "skipped_files": [(str(path), reason) for path, reason in skipped_files],
        "event_id_map": proc.event_id_map,
    }
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output_model)
    print(f"Saved model -> {args.output_model}")


if __name__ == "__main__":
    main()
