import collections
import os
import socket
import subprocess
import time

import numpy as np

from experiment_launcher import Launcher
from experiment_launcher.utils import is_local, get_slurm_jobs_in_queue
from mpd.utils.githash import get_git_hash_short


# ask for user permission
response = input(
    "This program might overwrite the data in data_trajectories folder." " Do you want to continue? (yes/no): "
).lower()
if response not in ["yes", "y"]:
    exit(1)


########################################################################################################################
# LAUNCHER

hostname = socket.gethostname()

LOCAL = is_local()
TEST = False
# USE_CUDA = True
USE_CUDA = False


N_EXPS_IN_PARALLEL = os.cpu_count() if not USE_CUDA and LOCAL else 1

# N_CORES = N_EXPS_IN_PARALLEL
N_CORES = 1
MEMORY_SINGLE_JOB = 3000
MEMORY_PER_CORE = N_EXPS_IN_PARALLEL * MEMORY_SINGLE_JOB // N_CORES
if "logc" in hostname:
    PARTITION = None
else:
    PARTITION = "gpu" if USE_CUDA else "amd,amd2"
GRES = "gpu:1" if USE_CUDA else None
CONDA_ENV = "mpd-splines"


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

MAX_SLURM_JOBS_IN_QUEUE = 400

########################################################################################################################
# EXPERIMENT PARAMETERS SETUP

exp_config = collections.namedtuple(
    "config",
    "env_id robot_id"
    " num_tasks num_trajectories_per_task"
    " min_distance_robot_env"
    " parametric_trajectory planner_allowed_time"
    " bspline_num_control_points bspline_degree"
    " sample_joint_position_goals_with_same_ee_pose"
    " cfg_file",
)

PLANNER = "RRTConnect"
# PLANNER = 'AITstar'

configs_d = {
    # many trajectories per task
    "many": [
        # joint to joint
        exp_config("EnvSimple2D", "RobotPointMass2D", 1000, 25, 0.01, PLANNER, 10.0, 30, 5, False, None),
        exp_config("EnvNarrowPassageDense2D", "RobotPointMass2D", 1000, 25, 0.01, PLANNER, 10.0, 30, 5, False, None),
        exp_config("EnvPlanar2Link", "RobotPlanar2Link", 1000, 25, 0.01, PLANNER, 10.0, 30, 5, False, None),
        exp_config("EnvPlanar4Link", "RobotPlanar4Link", 10000, 10, 0.01, PLANNER, 10.0, 30, 5, False, None),
        exp_config("EnvSpheres3D", "RobotPanda", 100000, 10, 0.02, PLANNER, 10.0, 38, 5, False, None),
        # joint or EE pose to EE pose with configuration file
        exp_config(
            "EnvWarehouse",
            "RobotPanda",
            500000,
            1,
            0.02,
            PLANNER,
            10.0,
            29,
            5,
            False,
            "EnvWarehouse-RobotPanda_v01.yaml",
        ),
    ],
    # one trajectory per task
    "one": [
        # joint to joint
        exp_config("EnvSimple2D", "RobotPointMass2D", 10000, 1, 0.01, PLANNER, 10.0, 30, 5, False, None),
        exp_config("EnvNarrowPassageDense2D", "RobotPointMass2D", 10000, 1, 0.01, PLANNER, 10.0, 30, 5, False, None),
        exp_config("EnvPlanar2Link", "RobotPlanar2Link", 10000, 1, 0.01, PLANNER, 10.0, 30, 5, False, None),
        exp_config("EnvPlanar4Link", "RobotPlanar4Link", 100000, 1, 0.01, PLANNER, 10.0, 30, 5, False, None),
        exp_config("EnvSpheres3D", "RobotPanda", 1000000, 1, 0.02, PLANNER, 10.0, 38, 5, False, None),
        exp_config("EnvWarehouse", "RobotPanda", 1000000, 1, 0.02, PLANNER, 10.0, 30, 5, False, None),
        exp_config("EnvPilars3D", "RobotPanda", 500000, 1, 0.02, PLANNER, 10.0, 30, 5, False, None),
        # with configuration file
        exp_config(
            "EnvWarehouse",
            "RobotPanda",
            500000,
            1,
            0.02,
            PLANNER,
            10.0,
            29,
            5,
            False,
            "EnvWarehouse-RobotPanda_v01.yaml",
        ),
    ],
}

configs_d_keys_filter = [
    # 'many',
    "one",
]

for selection in configs_d_keys_filter:

    configs = configs_d[selection]

    if selection == "many":
        N_TASKS_PER_EXPERIMENT = 100
    else:
        N_TASKS_PER_EXPERIMENT = 500

    ###############################
    # Launch jobs
    n_job = 0
    for k, config in enumerate(configs):
        n_tasks_to_process_total = config.num_tasks
        start_task_id = 0

        while n_tasks_to_process_total > 0:
            if not LOCAL:  # Wait for jobs to finish in the cluster
                while get_slurm_jobs_in_queue() >= MAX_SLURM_JOBS_IN_QUEUE:
                    sleep_seconds = 30
                    print(f"Waiting for jobs to finish. Sleeping for {sleep_seconds} seconds.")
                    time.sleep(sleep_seconds)

            # Launch new jobs
            n_tasks_to_process_in_experiment = N_TASKS_PER_EXPERIMENT
            if n_tasks_to_process_in_experiment > n_tasks_to_process_total:
                n_tasks_to_process_in_experiment = n_tasks_to_process_total

            print(
                f"---------> Launched job {n_job:4d} -- {config.env_id} and {config.robot_id} "
                f"for tasks {start_task_id}-{start_task_id + n_tasks_to_process_in_experiment}"
            )

            time.sleep(2)

            exp_name = f"{config.env_id}-{config.robot_id}"
            if config.cfg_file is not None:
                exp_name += f"-config_file"
            if config.sample_joint_position_goals_with_same_ee_pose:
                exp_name += "-joint_ee"
            else:
                exp_name += "-joint_joint"
            exp_name += f"-{selection}"
            exp_name += f"-{config.parametric_trajectory}"

            launcher = Launcher(
                exp_name=exp_name,
                exp_file="generate_trajectories",
                project_name="project02390",
                start_seed=start_task_id,
                n_seeds=1,
                n_exps_in_parallel=N_EXPS_IN_PARALLEL,
                n_cores=N_CORES if config.num_trajectories_per_task > 1 else 1,
                memory_per_core=MEMORY_PER_CORE,
                days=0,
                hours=23,
                minutes=59,
                seconds=0,
                partition=PARTITION,
                conda_env=CONDA_ENV,
                gres=GRES,
                use_timestamp=False,
                check_results_directories=False,
                compact_dirs=True,
                base_dir="../../data_trajectories",
            )

            ############################################################################################################
            # RUN
            extra_cfg_file = {}
            if config.cfg_file is not None:
                extra_cfg_file["cfg_file"] = config.cfg_file
                extra_cfg_file["configs__"] = "yes"
            else:
                extra_cfg_file["cfg_file"] = "None"

            print(f"extra_cfg_file: {extra_cfg_file}")

            launcher.add_experiment(
                env_id__=config.env_id,
                robot_id__=config.robot_id,
                **extra_cfg_file,
                sample_joint_position_goals_with_same_ee_pose__=config.sample_joint_position_goals_with_same_ee_pose,
                selection__=selection,
                planner__=config.parametric_trajectory,
                start_task_id=start_task_id,
                num_tasks=n_tasks_to_process_in_experiment,
                num_trajectories_per_task=config.num_trajectories_per_task,
                min_distance_robot_env=config.min_distance_robot_env,
                simplify_path=True,
                planner_allowed_time=config.planner_allowed_time,
                fit_bspline=False,  # do not fit a bspline during trajectory generation
                bspline_num_control_points=config.bspline_num_control_points,
                bspline_degree=config.bspline_degree,
                bspline_zero_vel_at_start_and_goal=True,
                bspline_zero_acc_at_start_and_goal=True,
                n_parallel_jobs=N_CORES,
                device="cuda" if USE_CUDA else "cpu",
                debug=False,
            )

            launcher.run(LOCAL, TEST)

            ############################################################################################################
            # Update counters
            start_task_id += n_tasks_to_process_in_experiment
            n_tasks_to_process_total -= n_tasks_to_process_in_experiment
            n_job += 1
