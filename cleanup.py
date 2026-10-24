
from pathlib import Path
import shutil

PROJECT = Path(r"C:\GarbageCNN").resolve()

# Experiment-specific model folders
MODEL_FOLDERS = [
    "models/v2",
    "models/v2_1",
    "models/v3",
    "models/v4",
]

# Experiment-specific result folders
RESULT_FOLDERS = [
    "results/v2",
    "results/v2_1",
    "results/v3",
    "results/v4",
]

# Experiment training scripts
TRAINING_SCRIPTS = [
    "trainv2.py",
    "trainv2_1.py",
    "trainv3.py",
    "trainv4.py",
]

# Optional V2–V4 experiment log files in the project root.
# Only include files explicitly named for those experiments.
ROOT_EXPERIMENT_FILES = []

TARGETS = (
    MODEL_FOLDERS
    + RESULT_FOLDERS
    + TRAINING_SCRIPTS
    + ROOT_EXPERIMENT_FILES
)


def safe_target(relative_path):
    target = (PROJECT / relative_path).resolve()

    # Never allow a target outside the project.
    if target != PROJECT and PROJECT not in target.parents:
        raise ValueError(f"Unsafe target rejected: {target}")

    # Protect the dataset, V1 artifacts, and dataset audit reports.
    protected = [
        PROJECT / "data",
        PROJECT / "models" / "garbage_cnn.keras",
        PROJECT / "results" / "dataset_audit",
        PROJECT / "train.py",
        PROJECT / "results" / "classification_report.txt",
        PROJECT / "results" / "classification_report.json",
        PROJECT / "results" / "confusion_matrix.png",
        PROJECT / "results" / "training_curves.png",
        PROJECT / "results" / "training_history.csv",
        PROJECT / "results" / "training_history_full.csv",
    ]

    for item in protected:
        item = item.resolve()

        if target == item or item in target.parents:
            raise ValueError(f"Protected path rejected: {target}")

    return target


def main():
    if not PROJECT.is_dir():
        raise FileNotFoundError(f"Project folder not found: {PROJECT}")

    existing = []

    print("GARBAGECNN EXPERIMENT CLEANUP")
    print(f"Project: {PROJECT}\n")
    print("The following existing targets will be considered:\n")

    for relative_path in TARGETS:
        target = safe_target(relative_path)

        if target.exists():
            existing.append((relative_path, target))
            print(f"  [DELETE] {relative_path}")

    if not existing:
        print("\nNo matching experiment files or folders were found.")
        return

    print("\nProtected: V1 model, V1 reports, dataset, dataset audit.")
    print("Review the list above before continuing.")

    confirmation = input(
        "\nType DELETE V2 V3 V4 to permanently delete these targets: "
    ).strip()

    if confirmation != "DELETE V2 V3 V4":
        print("Cancelled. Nothing was deleted.")
        return

    for relative_path, target in existing:
        if target.is_dir():
            shutil.rmtree(target)
        elif target.is_file():
            target.unlink()

        print(f"[DELETED] {relative_path}")

    print("\nCleanup complete.")
    print("V1 and the dataset audit were not targeted.")


if __name__ == "__main__":
    main()
