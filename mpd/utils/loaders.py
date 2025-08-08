import math
import os
from pathlib import Path
from typing import Union

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from mpd import models, losses, summaries
from mpd.datasets.trajectories_dataset_bspline import adjust_bspline_number_control_points, TrajectoryDatasetBspline
from mpd.datasets.trajectories_dataset_waypoints import TrajectoryDatasetWaypoints, adjust_waypoints
from mpd.parametric_trajectory.trajectory_bspline import ParametricTrajectoryBspline
from mpd.parametric_trajectory.trajectory_waypoints import ParametricTrajectoryWaypoints
from mpd.paths import DATASET_BASE_DIR
from mpd.utils import model_loader
from torch_robotics import environments, robots
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.torch_utils.torch_utils import freeze_torch_model_params, DEFAULT_TENSOR_ARGS


def get_planning_task_and_dataset(
    parametric_trajectory_class=None,
    phase_time_class="PhaseTimeLinear",
    phase_time_args={},
    dataset_subdir=None,
    dataset_file_merged="dataset_merged.hdf5",
    context_qs=False,
    context_ee_goal_pose=False,
    batch_size=2,
    val_set_size_fraction=0.025,
    results_dir=None,
    save_indices=False,
    load_indices=False,
    model_dir=None,
    preload_data_to_device=False,
    dataloader_num_workers=0,  # max(0, os.cpu_count() - 2),
    env_id_replace=False,
    obstacle_cutoff_margin_extra=0.0,
    margin_for_dense_collision_checking=0.0,
    grasped_object=None,
    # B-spline trajectory parameters
    bspline_degree=5,
    bspline_num_control_points_desired=13,
    bspline_zero_vel_at_start_and_goal=True,
    bspline_zero_acc_at_start_and_goal=True,
    num_T_pts=128,
    trajectory_duration=5.0,  # seconds
    tensor_args=DEFAULT_TENSOR_ARGS,
    **kwargs,
):
    dataset_subdir = dataset_subdir
    base_dir = os.path.join(DATASET_BASE_DIR, dataset_subdir)

    # get the dataset arguments
    dataset_args = load_params_from_yaml(os.path.join(base_dir, "args.yaml"))

    ##############################################################################################
    # -------------------------------- Load environment, robot, task -----------------------------
    # Environment
    env_class = getattr(environments, env_id_replace if env_id_replace else dataset_args["env_id"])
    env = env_class(**kwargs, tensor_args=tensor_args)

    # Robot
    robot_class = getattr(robots, dataset_args["robot_id"])
    # optionally attach a grasped object
    if grasped_object is not None and grasped_object.name is not None:
        grasped_object_class = getattr(environments, grasped_object.name)
        kwargs.update(
            grasped_object=grasped_object_class(**grasped_object, tensor_args=tensor_args),
        )
    robot = robot_class(**kwargs, tensor_args=tensor_args)

    # Task
    dataset_args["obstacle_cutoff_margin"] = dataset_args["min_distance_robot_env"] + obstacle_cutoff_margin_extra
    dataset_args["margin_for_dense_collision_checking"] = margin_for_dense_collision_checking

    # Create a parametric trajectory
    # Adjust the number of control points to match the Unet architecture.
    bspline_n_control_points, bspline_n_removed_control_points = adjust_bspline_number_control_points(
        bspline_num_control_points_desired,
        context_qs,
        context_ee_goal_pose,
        bspline_zero_vel_at_start_and_goal,
        bspline_zero_acc_at_start_and_goal,
    )
    print(f"--------------- Parametric trajectory -- {parametric_trajectory_class}")
    print(
        f"Number of B-spline control points.\n"
        f"\tdesired          : {bspline_num_control_points_desired}\n"
        f"\tadjusted         : {bspline_n_control_points}\n"
        f"\tlearnable + fixed: {bspline_n_control_points - bspline_n_removed_control_points} + {bspline_n_removed_control_points}\n"
    )

    if "ParametricTrajectoryBspline" in parametric_trajectory_class:
        # Construct the B-spline trajectory
        parametric_trajectory = ParametricTrajectoryBspline(
            n_control_points=bspline_n_control_points,
            degree=bspline_degree,
            zero_vel_at_start_and_goal=bspline_zero_vel_at_start_and_goal,
            zero_acc_at_start_and_goal=bspline_zero_acc_at_start_and_goal,
            num_T_pts=num_T_pts,
            remove_outer_control_points=context_qs,
            keep_last_control_point=context_ee_goal_pose,
            trajectory_duration=trajectory_duration,
            phase_time_class=phase_time_class,
            phase_time_args=phase_time_args,
            tensor_args=tensor_args,
        )
        dataset_class = TrajectoryDatasetBspline

    elif "ParametricTrajectoryWaypoints" in parametric_trajectory_class:
        # Adjust the number of waypoints to match the Unet architecture.
        # The number of learned waypoints should match the number of learned bspline control points (fair comparison)
        waypoint_num_points_desired = bspline_n_control_points - bspline_n_removed_control_points
        waypoint_num_points, n_removed_waypoints = adjust_waypoints(
            waypoint_num_points_desired, context_qs, context_ee_goal_pose
        )
        print(
            f"Number of (linear) waypoints.\n"
            f"\toriginal                   : {bspline_num_control_points_desired}\n"
            f"\tdesired (to match B-spline): {waypoint_num_points_desired}\n"
            f"\tadjusted                   : {waypoint_num_points}\n"
            f"\tlearnable + fixed          : {waypoint_num_points - n_removed_waypoints} + {n_removed_waypoints}\n"
        )

        parametric_trajectory = ParametricTrajectoryWaypoints(
            n_control_points=waypoint_num_points,
            remove_outer_control_points=context_qs,
            keep_last_control_point=context_ee_goal_pose,
            num_T_pts=num_T_pts,
            trajectory_duration=trajectory_duration,
            phase_time_class=phase_time_class,
            phase_time_args=phase_time_args,
            tensor_args=tensor_args,
        )
        dataset_class = TrajectoryDatasetWaypoints
    else:
        raise ValueError(f"Unknown parametric trajectory class: {parametric_trajectory_class}")

    planning_task = PlanningTask(
        env=env,
        robot=robot,
        parametric_trajectory=parametric_trajectory,
        **dataset_args,
        tensor_args=tensor_args,
    )

    ######################################################################################
    # -------------------------- Load dataset and create dataloaders ---------------------
    print("\n--------------- Loading data")
    full_dataset = dataset_class(
        planning_task=planning_task,
        base_dir=base_dir,
        dataset_file_merged=dataset_file_merged,
        preload_data_to_device=preload_data_to_device,
        context_qs=context_qs,
        context_ee_goal_pose=context_ee_goal_pose,
        tensor_args=tensor_args,
        **kwargs,
    )
    print(full_dataset)

    if load_indices:
        # load the indices of training and validation sets (for evaluation)
        assert model_dir is not None, "model_dir must be provided when load_indices is True"
        train_subset_indices = torch.load(os.path.join(model_dir, f"train_subset_indices.pt"))
        val_subset_indices = torch.load(os.path.join(model_dir, f"val_subset_indices.pt"))
    else:
        # split into train and validation
        # the training and validation sets are split by task id
        task_ids = list(full_dataset.map_task_id_to_control_points_id.keys())
        val_set_size = math.ceil(len(task_ids) * val_set_size_fraction)
        train_subset_task_indices, val_subset_task_indices = train_test_split(task_ids, test_size=val_set_size)
        train_subset_indices = np.concatenate(
            [
                full_dataset.map_task_id_to_control_points_id[train_subset_task_idx]
                for train_subset_task_idx in train_subset_task_indices
            ]
        )
        val_subset_indices = np.concatenate(
            [
                full_dataset.map_task_id_to_control_points_id[val_subset_task_idx]
                for val_subset_task_idx in val_subset_task_indices
            ]
        )
    train_subset = torch.utils.data.Subset(full_dataset, train_subset_indices)
    val_subset = torch.utils.data.Subset(full_dataset, val_subset_indices)

    assert len(train_subset.indices) > 0, "train_subset cannot be empty"
    print(f"train_subset size: {len(train_subset.indices)}")
    assert len(val_subset.indices) > 0, "val_subset cannot be empty"
    print(f"val_subset size  : {len(val_subset.indices)}")
    data_loader_options = {}
    if not preload_data_to_device:
        data_loader_options["num_workers"] = dataloader_num_workers
        data_loader_options["pin_memory"] = True
        data_loader_options["persistent_workers"] = True if dataloader_num_workers > 0 else False
    train_dataloader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, drop_last=False, **data_loader_options
    )
    val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=True, drop_last=False, **data_loader_options)

    if not load_indices and save_indices:
        # save the indices of training and validation sets (for later evaluation)
        torch.save(train_subset.indices, os.path.join(results_dir, f"train_subset_indices.pt"))
        torch.save(val_subset.indices, os.path.join(results_dir, f"val_subset_indices.pt"))

    return planning_task, train_subset, train_dataloader, val_subset, val_dataloader


@model_loader
def get_model(
    model_class=None, checkpoint_path=None, freeze_loaded_model=False, tensor_args=DEFAULT_TENSOR_ARGS, **kwargs
):

    if checkpoint_path is not None:
        model = torch.load(checkpoint_path)
        if freeze_loaded_model:
            freeze_torch_model_params(model)
    else:
        ModelClass = getattr(models, model_class)
        model = ModelClass(**kwargs).to(tensor_args["device"])

    return model


def build_module(model_class=None, submodules=None, **kwargs):
    if submodules is not None:
        for key, value in submodules.items():
            kwargs[key] = build_module(**value)

    Model = getattr(models, model_class)
    model = Model(**kwargs)

    return model


def get_loss(loss_class=None, **kwargs):
    LossClass = getattr(losses, loss_class)
    loss = LossClass(**kwargs)
    loss_fn = loss.loss_fn
    return loss_fn


def get_summary(summary_class=None, **kwargs):
    if summary_class is None:
        return None
    SummaryClass = getattr(summaries, summary_class)
    summary_fn = SummaryClass(**kwargs).summary_fn
    return summary_fn


def load_params_from_yaml(path: Union[str, Path]) -> dict:
    with open(path, "r") as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def save_to_yaml(data: dict, path):
    with open(path, "w") as stream:
        yaml.dump(data, stream, Dumper=yaml.Dumper, allow_unicode=True)
