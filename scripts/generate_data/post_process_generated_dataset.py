import isaacgym
import argparse
import glob
import os.path

import h5py
import numpy as np
import psutil
import yaml
from joblib import Parallel, delayed

from mpd.utils.loaders import load_params_from_yaml
from torch_robotics.torch_utils.torch_timer import TimerCUDA


if __name__ == "__main__":
    #################################################################################
    parser = argparse.ArgumentParser(description="Post process logs directory")
    parser.add_argument(
        "--data_dir",
        type=str,
        # default='../../data_trajectories/EnvSimple2D-RobotPointMass2D-many',
        default="./data/env-robot/",
        help="Top-level directory containing the raw data generated",
    )
    cli_args = parser.parse_args()

    # -------------------------------- Join data -------------------------
    # Join all the datasets into one in a non-optimal way (memory inefficient)
    # It reads all the h5 files to memory, creates a dictionary with all the datasets, and writes to a new h5 file

    def print_attrs(name, obj):
        print(name)
        for key, val in obj.attrs.items():
            print("    %s: %s" % (key, val))

    def read_one_dataset(dataset_h5_file_path):
        dataset_dict = {}
        h5fr = h5py.File(dataset_h5_file_path, "r")
        # h5fr.visititems(print_attrs)
        for k in h5fr.keys():
            dataset_dict[k] = h5fr[k][:]
        dataset_dict["num_trajectories_desired"] = h5fr.attrs["num_trajectories_desired"]
        dataset_dict["num_trajectories_generated"] = h5fr.attrs["num_trajectories_generated"]
        h5fr.close()
        return dataset_dict

    # get the results from all the datasets
    dataset_h5_files = glob.glob(f"{cli_args.data_dir}/**/dataset.hdf5", recursive=True)
    print("#################################################################################")
    print("\nGetting datasets into a list of dictionaries...")
    with TimerCUDA() as t:
        dataset_dict_l = Parallel(n_jobs=-1)(
            delayed(read_one_dataset)(dataset_h5_file) for dataset_h5_file in dataset_h5_files
        )
    print(f"Elapsed time: {t.elapsed:.2f} sec")
    print(f"Used memory: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MiB")

    # merge all the datasets into one
    num_trajectories_desired = 0
    num_trajectories_generated = 0
    print("\nMerging list of datasets into one dictionary...")
    with TimerCUDA() as t:
        # Allocate memory for the dictionary. This is much faster than concatenating inside the loop
        dataset_all_dict_shapes_dtype = {}
        for dataset_dict in dataset_dict_l:
            for k, v in dataset_dict.items():
                if k == "num_trajectories_desired":
                    num_trajectories_desired += v
                    continue
                elif k == "num_trajectories_generated":
                    num_trajectories_generated += v
                    continue

                if k in dataset_all_dict_shapes_dtype:
                    # increment the batch dimension
                    dataset_all_dict_shapes_dtype[k]["shape"][0] += v.shape[0]
                else:
                    dataset_all_dict_shapes_dtype[k] = {"shape": list(v.shape), "dtype": v.dtype}

        dataset_all_dict = {}
        for k, v in dataset_all_dict_shapes_dtype.items():
            dataset_all_dict[k] = np.empty(v["shape"], dtype=v["dtype"])

        # merge the datasets
        idxs = {}
        for i, dataset_dict in enumerate(dataset_dict_l):
            if i == 0 or i % 10 == 0 or i == len(dataset_dict_l) - 1:
                print(f".........Merging dataset {i}/{len(dataset_dict_l)-1}")
            for k, v in dataset_dict.items():
                if k == "num_trajectories_desired" or k == "num_trajectories_generated":
                    continue

                if k in idxs:
                    pass
                else:
                    idxs[k] = 0
                dataset_all_dict[k][idxs[k] : idxs[k] + v.shape[0]] = v
                idxs[k] += v.shape[0]

    print(f"Elapsed time: {t.elapsed:.2f} sec")
    print(f"Used memory: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MiB")

    # write to one hdf5 file
    print("\nWriting to one hdf5 file...")
    with TimerCUDA() as t:
        hf = h5py.File(os.path.join(cli_args.data_dir, "dataset_merged.hdf5"), mode="w")
        for i, (k, v) in enumerate(dataset_all_dict.items()):
            if i == 0 or i % 2 == 0:
                print(f".........Writing dataset key {i}/{len(dataset_all_dict)}")
            hf.create_dataset(k, data=v)
    print(f"Elapsed time: {t.elapsed:.2f} sec")
    print(f"Used memory: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MiB")

    # Statistics
    print(
        f"\nNumber of trajectories generated/desired: {num_trajectories_generated}/{num_trajectories_desired} "
        f"({num_trajectories_generated/num_trajectories_desired * 100:.2f}%)"
    )

    hf.close()

    # save one args file to the new directory
    args_file = glob.glob(f"{cli_args.data_dir}/**/args.yaml", recursive=True)[0]
    args_data = load_params_from_yaml(args_file)

    # remove some keys
    args_data.pop("num_tasks", None)
    args_data.pop("results_dir", None)
    args_data.pop("seed", None)
    args_data.pop("start_task_id", None)

    args_data["num_trajectories_desired"] = int(num_trajectories_desired)
    args_data["num_trajectories_generated"] = int(num_trajectories_generated)

    # save args file
    with open(os.path.join(cli_args.data_dir, "args.yaml"), "w") as outfile:
        yaml.dump(args_data, outfile, default_flow_style=False)
