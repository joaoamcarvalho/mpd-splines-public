import os
from collections import OrderedDict
from itertools import product
from pathlib import Path

from experiment_launcher import Launcher
from experiment_launcher.utils import is_local

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

########################################################################################################################
# LAUNCHER

LOCAL = is_local()
USE_CUDA = True

N_EXPS_IN_PARALLEL = 1

N_CORES = N_EXPS_IN_PARALLEL * 2
MEMORY_SINGLE_JOB = 6000
PARTITION = "gpu" if USE_CUDA else "amd3,amd2,amd"
GRES = "gpu:1" if USE_CUDA else None
CONSTRAINT = "rtx3090|a5000" if USE_CUDA else None  # rtx2080|rtx3080|rtx3090|a5000
CONDA_ENV = "mpd-splines"


EXPERIMENT_NAME = Path(__file__).stem

launcher = Launcher(
    exp_name=EXPERIMENT_NAME,
    exp_file="train",
    n_seeds=1,
    n_exps_in_parallel=N_EXPS_IN_PARALLEL,
    n_cores=N_CORES,
    memory_per_core=N_EXPS_IN_PARALLEL * MEMORY_SINGLE_JOB // N_CORES,
    days=2,
    hours=23,
    minutes=59,
    seconds=0,
    partition=PARTITION,
    conda_env=CONDA_ENV,
    gres=GRES,
    constraint=CONSTRAINT,
    use_timestamp=True,
)

########################################################################################################################
# Setup and run experiments
single_experiment_params_base_l = [
    # EnvSimple2D-RobotPointMass2D
    OrderedDict(
        dataset_subdir__="EnvSimple2D-RobotPointMass2D-joint_joint-one-RRTConnect",
        context_ee_goal_pose__=False,
        batch_size=128,
        num_train_steps=2_000_000,
        unet_input_dim__=32,
        unet_dim_mults_option__=0,
        context_qs_n_layers=1,
        context_q_out_dim__=32,
        context_ee_goal_pose_n_layers=1,
        context_ee_goal_pose_out_dim__=32,
        context_combined_out_dim__=32,
        bspline_num_control_points_desired__=22,
    ),
    # EnvNarrowPassageDense2D-RobotPointMass2D
    OrderedDict(
        dataset_subdir__="EnvNarrowPassageDense2D-RobotPointMass2D-joint_joint-one-RRTConnect",
        context_ee_goal_pose__=False,
        batch_size=128,
        num_train_steps=2_000_000,
        unet_input_dim__=32,
        unet_dim_mults_option__=0,
        context_qs_n_layers=1,
        context_q_out_dim__=32,
        context_ee_goal_pose_n_layers=1,
        context_ee_goal_pose_out_dim__=32,
        context_combined_out_dim__=32,
        bspline_num_control_points_desired__=30,
    ),
    # EnvPlanar2Link-RobotPlanar2Link
    OrderedDict(
        dataset_subdir__="EnvPlanar2Link-RobotPlanar2Link-joint_joint-one-RRTConnect",
        context_ee_goal_pose__=False,
        batch_size=128,
        num_train_steps=2_000_000,
        unet_input_dim__=32,
        unet_dim_mults_option__=0,
        context_qs_n_layers=1,
        context_q_out_dim__=32,
        context_ee_goal_pose_n_layers=1,
        context_ee_goal_pose_out_dim__=32,
        context_combined_out_dim__=32,
        bspline_num_control_points_desired__=22,
    ),
    # EnvPlanar4Link-RobotPlanar4Link
    OrderedDict(
        dataset_subdir__="EnvPlanar4Link-RobotPlanar4Link-joint_joint-one-RRTConnect",
        context_ee_goal_pose__=True,
        batch_size=512,
        num_train_steps=3_000_000,
        unet_input_dim__=32,
        unet_dim_mults_option__=1,
        context_qs_n_layers=2,
        context_q_out_dim__=128,
        context_ee_goal_pose_n_layers=2,
        context_ee_goal_pose_out_dim__=128,
        context_combined_out_dim__=128,
        bspline_num_control_points_desired__=22,
    ),
    # EnvSpheres3D-RobotPanda
    OrderedDict(
        dataset_subdir__="EnvSpheres3D-RobotPanda-joint_joint-one-RRTConnect",
        context_ee_goal_pose__=True,
        batch_size=512,
        num_train_steps=3_000_000,
        unet_input_dim__=32,
        unet_dim_mults_option__=1,
        context_qs_n_layers=2,
        context_q_out_dim__=128,
        context_ee_goal_pose_n_layers=2,
        context_ee_goal_pose_out_dim__=128,
        context_combined_out_dim__=128,
        bspline_num_control_points_desired__=30,
    ),
    # EnvWarehouse-RobotPanda - config_file
    OrderedDict(
        dataset_subdir__="EnvWarehouse-RobotPanda-config_file_v01-joint_joint-one-RRTConnect",
        context_ee_goal_pose__=True,
        batch_size=512,
        num_train_steps=3_000_000,
        unet_input_dim__=32,
        unet_dim_mults_option__=1,
        context_qs_n_layers=2,
        context_q_out_dim__=128,
        context_ee_goal_pose_n_layers=2,
        context_ee_goal_pose_out_dim__=128,
        context_combined_out_dim__=128,
        bspline_num_control_points_desired__=22,
    ),
]

parametric_trajectory_class_l = []
parametric_trajectory_class_l += [
    OrderedDict(parametric_trajectory_class__="ParametricTrajectoryBspline"),
    OrderedDict(parametric_trajectory_class__="ParametricTrajectoryWaypoints"),
]

dataset_file_merged_l = [
    OrderedDict(dataset_file_merged__="dataset_merged_doubled.hdf5"),
]

single_experiment_params_l = []
for single_experiment_params_base, parametric_trajectory_class, dataset_file_merged in product(
    single_experiment_params_base_l, parametric_trajectory_class_l, dataset_file_merged_l
):
    single_experiment_params = single_experiment_params_base.copy()
    single_experiment_params.update(parametric_trajectory_class)

    # trick to reorder the keys
    single_experiment_params.move_to_end("dataset_subdir__")
    single_experiment_params.update(dataset_file_merged)
    single_experiment_params.move_to_end("dataset_file_merged__", last=False)
    single_experiment_params.move_to_end("dataset_subdir__", last=False)

    single_experiment_params_l.append(single_experiment_params)


os.environ["WANDB_API_KEY"] = "999"
wandb_options = dict(
    wandb_mode="online", wandb_entity="mpd-splines", wandb_project=EXPERIMENT_NAME  # "online", "offline" or "disabled"
)

for single_experiment_params in single_experiment_params_l:

    wandb_run_name_l = []
    for k, v in single_experiment_params.items():
        if k.endswith("__"):
            wandb_run_name_l += [f"{v}"]
    wandb_run_name = "-".join(wandb_run_name_l)
    wandb_run_name = wandb_run_name.replace(".hdf5", "")
    wandb_run_name = wandb_run_name.replace("ParametricTrajectory", "")
    wandb_run_name = wandb_run_name[:127]  # wandb run name has a limit of 128 characters
    wandb_options.update(
        wandb_run_name=wandb_run_name,
    )
    launcher.add_experiment(
        generative_model_class__="GaussianDiffusionModel",
        **single_experiment_params,
        context_qs=True,
        reload_data=False,
        preload_data_to_device=False,
        n_task_samples=-1,
        clip_grad=True,
        use_ema=True,
        steps_til_summary=25_000,
        steps_til_ckpt=50_000,
        device="cuda:0",
        **wandb_options,
        debug=False,
    )

launcher.run(LOCAL)
