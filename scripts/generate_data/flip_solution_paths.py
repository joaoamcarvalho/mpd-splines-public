"""
Doubles the dataset size by flipping the solution paths in the dataset to augment the dataset.
"""

from copy import copy
import h5py
from tqdm import tqdm
import numpy as np
import glob
import gc
import os
from joblib import Parallel, delayed


def create_doubled_dataset(f_path):
    print(f_path)
    hf_target_path = f_path.replace("dataset_merged.hdf5", "dataset_merged_doubled.hdf5")
    if os.path.exists(hf_target_path):
        return

    dataset_h5 = h5py.File(f_path, "r")
    dataset_all_dict = {k: [] for k in dataset_h5.keys()}
    max_task_id = max(dataset_h5["task_id"])
    for i in tqdm(range(len(dataset_h5["sol_path"]))):
        for k in dataset_h5.keys():
            if k == "sol_path":
                sol_path = dataset_h5[k][i]
                # flip the solution path, so that the start and end points are swapped
                sol_path_flipped = copy(sol_path[::-1])
                dataset_all_dict[k].append(sol_path)
                dataset_all_dict[k].append(sol_path_flipped)
            elif k == "task_id":
                dataset_all_dict[k].append(dataset_h5[k][i])
                dataset_all_dict[k].append(dataset_h5[k][i] + max_task_id + 1)  # add a new task_id
            else:
                # append data twice - for the original and flipped solution paths
                dataset_all_dict[k].append(dataset_h5[k][i])
                dataset_all_dict[k].append(dataset_h5[k][i])
    dataset_h5.close()
    gc.collect()

    hf = h5py.File(hf_target_path, mode="w")
    for i, (k, v) in enumerate(dataset_all_dict.items()):
        if i == 0 or i % 2 == 0:
            print(f".........Writing dataset key {i}/{len(dataset_all_dict)}")
        hf.create_dataset(k, data=v)
    hf.close()
    del dataset_all_dict
    gc.collect()


if __name__ == "__main__":
    # create datasets in parallel with joblib
    PATH_TO_DATASETS = "/home/carvalho/Projects/MotionPlanningDiffusion/mpd-splines/data_trajectories/EnvEmpty2D-RobotPointMass2D-joint_joint-many-GPPrior/**/*dataset_merged.hdf5"
    N_JOBS = 5
    Parallel(n_jobs=N_JOBS)(
        delayed(create_doubled_dataset)(f_path) for f_path in glob.glob(PATH_TO_DATASETS, recursive=True)
    )
