"""Shared helpers for the tutorial's platform jobs, plus a CLI.

Notebooks 2 and 6 import this module to register and run the jobs; from an
IDE the same flows run as `python run_job.py create|train|retrain [drop]`.
"""

import os
import sys

import hopsworks

TRAIN_JOB_NAME = "train_yolo"
CREATE_FGS_JOB_NAME = "create_fgs"


def tutorial_dir_project_path(project):
    """Project-relative path of this tutorial folder (must contain train.py).

    The job's appPath must reference the folder at its project (HDFS) path so
    train.py's sibling files (model.yaml, weights/, data/) are available at run
    time. With a mounted project filesystem the working directory ends with the
    folder's project-relative path, so probe cwd suffixes; fall back to the
    conventional locations when running from an IDE.
    """
    dataset_api = project.get_dataset_api()
    parts = os.getcwd().strip(os.sep).split(os.sep)
    candidates = ["/".join(parts[i:]) for i in range(len(parts))]
    candidates += ["Jupyter/yolov8-face", "Resources/yolov8-face"]
    for candidate in candidates:
        if candidate and dataset_api.exists(f"{candidate}/train.py"):
            return candidate
    raise RuntimeError(
        "Could not locate the yolov8-face folder (train.py) in the project "
        "filesystem. Clone/upload it into the project (e.g. into the Jupyter "
        "dataset) first."
    )


def _ensure_job(project, name, app_file, environment, cores, memory, gpus=0,
                recreate=False):
    job_api = project.get_job_api()
    job = job_api.get_job(name)
    if job is not None and not recreate:
        return job
    print(f"Registering job '{name}' ...")
    config = job_api.get_configuration("PYTHON")
    config["appPath"] = f"{tutorial_dir_project_path(project)}/{app_file}"
    config["environmentName"] = environment
    config["resourceConfig"]["cores"] = cores
    config["resourceConfig"]["memory"] = memory
    if gpus:
        config["resourceConfig"]["gpus"] = gpus
    return job_api.create_job(name, config)


def ensure_train_yolo_job(project, recreate=False):
    """Get or register the train_yolo job (train.py, yolov8 GPU env)."""
    return _ensure_job(project, TRAIN_JOB_NAME, "train.py", "yolov8",
                       cores=1, memory=10000, gpus=1, recreate=recreate)


def ensure_create_fgs_job(project, recreate=False):
    """Get or register the create_fgs job (create_fgs.py)."""
    return _ensure_job(project, CREATE_FGS_JOB_NAME, "create_fgs.py",
                       "pandas-training-pipeline", cores=1, memory=4096,
                       recreate=recreate)


def run_and_print_logs(job, args=None):
    """Run the job, wait for it to finish, and print its logs."""
    print(f"Running job '{job.name}'" + (f" with args '{args}'" if args else "") + " ...")
    execution = job.run(args=args, await_termination=True)
    out, err = execution.download_logs()
    for path in (out, err):
        if path:
            with open(path) as f:
                print(f.read())
    return execution


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    recreate = len(sys.argv) > 2 and sys.argv[2].lower() == "drop"

    if command not in ("create", "train", "retrain"):
        print("Usage: python run_job.py create|train|retrain [drop]")
        sys.exit(-1)

    project = hopsworks.login()
    if command == "create":
        job = ensure_create_fgs_job(project, recreate=recreate)
        run_and_print_logs(job)
    else:
        job = ensure_train_yolo_job(project, recreate=recreate)
        run_and_print_logs(job, args="retrain" if command == "retrain" else None)
