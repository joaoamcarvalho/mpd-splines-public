import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from matplotlib.collections import LineCollection

from mpd.summaries.summary_base import SummaryBase
from torch_robotics.trajectory.metrics import compute_ee_pose_errors
from torch_robotics.torch_utils.torch_utils import to_torch, dict_to_device, DEFAULT_TENSOR_ARGS


class SummaryTrajectoryGeneration(SummaryBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def summary_fn(
        self,
        train_step=None,
        model=None,
        datasubset=None,
        prefix="",
        debug=False,
        batch_size_statistics=10,
        planning_task=None,
        tensor_args=DEFAULT_TENSOR_ARGS,
        **kwargs,
    ):

        dataset = datasubset.dataset

        # ------------------------------------------------------------------------------------
        # Compute statistics on a set of random tasks
        # Get random tasks from the dataset
        control_points_idxs = np.random.choice(
            datasubset.indices, size=min(batch_size_statistics, len(datasubset.indices)), replace=False
        )

        n_samples = 25
        horizon = dataset.n_learnable_control_points
        control_points_normalized_l = []
        q_start_l = []
        q_goal_l = []
        ee_goal_pose_l = []
        for idx in control_points_idxs:
            data_sample = dict_to_device(dataset[idx], device=tensor_args["device"])

            q_start_l.append(einops.repeat(data_sample["q_start"], "... -> n ...", n=n_samples))
            q_goal_l.append(einops.repeat(data_sample["q_goal"], "... -> n ...", n=n_samples))
            ee_goal_pose_l.append(einops.repeat(data_sample["ee_goal_pose"], "... -> n ...", n=n_samples))

            # ------------------------------------------------------------------------------------
            # Sample control points with the inference model
            hard_conds = data_sample["hard_conds"]
            context_d = dataset.build_context(data_sample=data_sample)
            control_points_normalized_tmp = model.run_inference(
                context_d=context_d,
                hard_conds=hard_conds,
                n_samples=n_samples,
                horizon=horizon,
            )
            control_points_normalized_l.append(control_points_normalized_tmp)

        control_points_normalized = torch.cat(control_points_normalized_l, dim=0)

        # unnormalize control points samples
        control_points = dataset.unnormalize_control_points(control_points_normalized)

        # Get the bspline trajectory
        q_start = torch.cat(q_start_l, dim=0)
        q_goal = torch.cat(q_goal_l, dim=0)
        q_pos_trajs = planning_task.parametric_trajectory.get_q_trajectory(
            control_points, q_start, q_goal, get_type=["pos"]
        )["pos"]

        # ------------------------------------------------------------------------------------
        # STATISTICS
        wandb.log(
            {f"{prefix}percentage free trajs": planning_task.compute_fraction_valid_trajs(q_pos_trajs)}, step=train_step
        )
        wandb.log(
            {f"{prefix}percentage collision intensity": planning_task.compute_collision_intensity_trajs(q_pos_trajs)},
            step=train_step,
        )
        wandb.log({f"{prefix}success": planning_task.compute_success_valid_trajs(q_pos_trajs)}, step=train_step)

        # EE pose errors
        ee_pose_goal = torch.cat(ee_goal_pose_l, dim=0)
        ee_pose_goal_achieved = planning_task.robot.get_EE_pose(q_pos_trajs[..., -1, :])
        error_ee_pose_goal_position, error_ee_pose_goal_orientation = compute_ee_pose_errors(
            ee_pose_goal, ee_pose_goal_achieved
        )
        ee_pose_goal_error_position_norm = torch.linalg.norm(error_ee_pose_goal_position, dim=-1)
        ee_pose_goal_error_orientation_norm = torch.rad2deg(torch.linalg.norm(error_ee_pose_goal_orientation, dim=-1))

        wandb.log(
            {f"{prefix}ee_pose_goal_error_position_norm MEAN": ee_pose_goal_error_position_norm.mean()}, step=train_step
        )
        wandb.log(
            {f"{prefix}ee_pose_goal_error_position_norm STD": ee_pose_goal_error_position_norm.std()}, step=train_step
        )
        wandb.log(
            {f"{prefix}ee_pose_goal_error_orientation_norm MEAN": ee_pose_goal_error_orientation_norm.mean()},
            step=train_step,
        )
        wandb.log(
            {f"{prefix}ee_pose_goal_error_orientation_norm STD": ee_pose_goal_error_orientation_norm.std()},
            step=train_step,
        )

        # ------------------------------------------------------------------------------------
        # Render one task idx
        idx_render = np.random.choice(datasubset.indices)
        data_sample_render = dict_to_device(dataset[idx_render], device=tensor_args["device"])
        context_d_render = dataset.build_context(data_sample=data_sample_render)
        hard_conds_render = data_sample_render["hard_conds"]
        control_points_normalized = model.run_inference(
            context_d=context_d_render,
            hard_conds=hard_conds_render,
            n_samples=n_samples,
            horizon=horizon,
        )

        # unnormalize control points samples
        control_points = dataset.unnormalize_control_points(control_points_normalized)

        # Get the bspline trajectory
        q_start = einops.repeat(data_sample_render["q_start"], "d -> n d", n=n_samples)
        q_goal = einops.repeat(data_sample_render["q_goal"], "d -> n d", n=n_samples)
        q_trajs_d = planning_task.parametric_trajectory.get_q_trajectory(
            control_points, q_start, q_goal, get_type=("pos", "vel", "acc"), get_time_representation=True
        )
        q_pos_trajs = q_trajs_d["pos"]
        q_vel_trajs = q_trajs_d["vel"]
        q_acc_trajs = q_trajs_d["acc"]

        # render only some trajectories for planar robots
        render_n_robot_trajectories = 1 if "planar" in planning_task.robot.name.lower() else -1

        # dataset trajectory
        fig_joint_trajs_dataset, _, fig_robot_trajs_dataset, _ = dataset.render(
            task_id=dataset.map_control_points_id_to_task_id[idx_render],
            render_joint_trajectories=True,
            # render robot trajectories only for 2D robots
            render_robot_trajectories=True if planning_task.robot.task_space_dim == 2 else False,
            render_n_robot_trajectories=render_n_robot_trajectories,
        )
        fig_joint_trajs_dataset.suptitle("Dataset")

        # inferred trajectory
        fig_joint_trajs_generator, _ = planning_task.plot_joint_space_trajectories(
            q_pos_trajs=q_pos_trajs,
            q_vel_trajs=q_vel_trajs,
            q_acc_trajs=q_acc_trajs,
            q_pos_start=q_start[0],
            q_pos_goal=q_goal[0],
            set_q_pos_limits=True,
            set_q_vel_limits=True,
            set_q_acc_limits=True,
        )
        fig_joint_trajs_generator.suptitle("Generator")

        # robot trajectory (if 2D)
        fig_robot_trajs_generator = None
        if planning_task.robot.task_space_dim == 2:
            if "planar" in planning_task.robot.name.lower():
                q_pos_trajs = q_pos_trajs[
                    np.random.choice(q_pos_trajs.shape[0], render_n_robot_trajectories, replace=False)
                ]
                q_pos_trajs = q_pos_trajs[..., :: q_pos_trajs.shape[-2] // 10, :]

            ee_goal_position = data_sample_render[dataset.field_key_context_ee_goal_position]
            fig_robot_trajs_generator, _ = planning_task.render_robot_trajectories(
                q_pos_trajs=q_pos_trajs,
                q_pos_start=q_start[0],
                q_pos_goal=None if dataset.context_ee_goal_pose else q_goal[0],
                ee_goal_position=ee_goal_position,
                control_points=control_points,
            )

        # Log to wandb
        if fig_joint_trajs_dataset is not None:
            wandb.log({f"{prefix}joint trajectories DATASET": wandb.Image(fig_joint_trajs_dataset)}, step=train_step)
        if fig_robot_trajs_dataset is not None:
            wandb.log({f"{prefix}robot trajectories DATASET": wandb.Image(fig_robot_trajs_dataset)}, step=train_step)

        if fig_joint_trajs_generator is not None:
            wandb.log(
                {f"{prefix}joint trajectories GENERATOR": wandb.Image(fig_joint_trajs_generator)}, step=train_step
            )
        if fig_robot_trajs_generator is not None:
            wandb.log(
                {f"{prefix}robot trajectories GENERATOR": wandb.Image(fig_robot_trajs_generator)}, step=train_step
            )

        if debug:
            plt.show()

        if fig_joint_trajs_dataset is not None:
            plt.close(fig_joint_trajs_dataset)
        if fig_robot_trajs_dataset is not None:
            plt.close(fig_robot_trajs_dataset)
        if fig_joint_trajs_generator is not None:
            plt.close(fig_joint_trajs_generator)
        if fig_robot_trajs_generator is not None:
            plt.close(fig_robot_trajs_generator)
