import glob
import os

from typing import Any, Callable, Dict, Optional

# copied from stable-baselines3 zoo
def get_latest_run_id(log_path: str, run_name: str = "solver_train_") -> int:
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.

    :param log_path: path to log folder
    :return: latest run number
    """
    max_run_id = 0
    for path in glob.glob(os.path.join(log_path, run_name + "[0-9]*")):
        file_name = os.path.basename(path)
        ext = file_name.split("_")[-1]
        if ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id
