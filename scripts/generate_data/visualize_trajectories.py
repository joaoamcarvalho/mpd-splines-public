from pathlib import Path

import click

import isaacgym

from pprint import pprint

import h5py
import seaborn

from mpd.parametric_trajectory.trajectory_bspline import ParametricTrajectoryBspline
from mpd.paths import DATASET_BASE_DIR
from pb_ompl.pb_ompl import fit_bspline_to_path
from torch_robotics.isaac_gym_envs.motion_planning_envs import (
    MotionPlanningIsaacGymEnv,
    MotionPlanningControllerIsaacGym,
)
from scripts.generate_data.generate_trajectories import GenerateDataOMPL

import matplotlib.pyplot as plt

import os.path

import numpy as np
import torch
import yaml
from scipy import interpolate

from mpd.utils.loaders import load_params_from_yaml
from torch_robotics import environments, robots
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.torch_kinematics_tree.utils.files import get_robot_path
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, to_torch, to_numpy


@click.command()
@click.argument(
    "data_dir",
    type=click.Path(exists=True),
    default=Path(DATASET_BASE_DIR) / "EnvSimple2D-RobotPointMass2D-joint_joint-one-RRTConnect",
)
def visualize(data_dir):
    os.makedirs(os.path.join(data_dir, "figures"), exist_ok=True)

    fix_random_seed(3)

    isaac_gym_render_all_trajectories = False

    tensor_args = DEFAULT_TENSOR_ARGS
    tensor_args["device"] = "cpu"

    # -------------------------------- Load trajectories -------------------------
    n_tasks_display = 5  # plot this number of tasks
    n_trajs_display = 5  # plot this number of trajectories per task

    # get the args
    args = load_params_from_yaml(os.path.join(data_dir, "args.yaml"))
    print(f"\n-------------- ARGS --------------")
    print(yaml.dump(args))

    # get the merged dataset file
    dataset_h5 = h5py.File(os.path.join(data_dir, "dataset_merged_doubled.hdf5"), "r")

    # -------------------------------- Load env, robot, task ---------------------------------
    # Environment
    env_class = getattr(environments, args["env_id"])
    env = env_class(tensor_args=tensor_args)

    # Robot
    robot_class = getattr(robots, args["robot_id"])
    robot = robot_class(tensor_args=tensor_args, gripper=True)

    # Task
    parametric_trajectory = ParametricTrajectoryBspline(
        n_control_points=16,
        degree=5,
        num_T_pts=128,
        zero_vel_at_start_and_goal=True,
        zero_acc_at_start_and_goal=True,
        remove_outer_control_points=False,
        keep_last_control_point=False,
        trajectory_duration=5.0,
        tensor_args=tensor_args,
        phase_time_class="PhaseTimeLinear",
    )

    planning_task = PlanningTask(
        parametric_trajectory=parametric_trajectory,
        env=env,
        robot=robot,
        obstacle_cutoff_margin=args["min_distance_robot_env"],
        tensor_args=tensor_args,
    )

    # -------------------------------- Fit the b-spline ---------------------------------
    q_pos_trajs = []
    q_control_points = []

    task_ids_selected = np.random.choice(np.unique(dataset_h5["task_id"]), n_tasks_display, replace=False)
    for task_id_selected in task_ids_selected:
        idxs_with_task_id = np.argwhere(dataset_h5["task_id"] == task_id_selected).squeeze(1)

        for i in np.random.choice(idxs_with_task_id, min(n_trajs_display, len(idxs_with_task_id))):
            if "bspline_params_cc" in dataset_h5:
                bspline_params = (
                    dataset_h5["bspline_params_tt"][i],
                    dataset_h5["bspline_params_cc"][i],
                    dataset_h5["bspline_params_k"][i],
                )
            else:
                # fit a spline to the path
                bspline_params = fit_bspline_to_path(
                    dataset_h5["sol_path"][i],
                    bspline_degree=parametric_trajectory.bspline.d,
                    bspline_num_control_points=parametric_trajectory.bspline.n_pts,
                    bspline_zero_vel_at_start_and_goal=parametric_trajectory.zero_vel_at_start_and_goal,
                    bspline_zero_acc_at_start_and_goal=parametric_trajectory.zero_acc_at_start_and_goal,
                    debug=True,
                )

            tt, cc, k = bspline_params
            cc_np = np.array(cc)

            bspl = interpolate.BSpline(tt, cc_np.T, k)  # note the transpose
            interpolate_num = 128
            u_interpolation = np.linspace(0, 1, interpolate_num)
            bspline_path_interpolated = bspl(u_interpolation)

            q_pos_trajs.append(to_torch(bspline_path_interpolated, **tensor_args))
            q_control_points.append(to_torch(cc_np.T, **tensor_args))

    q_control_points = torch.stack(q_control_points)
    q_pos_trajs = torch.stack(q_pos_trajs)

    # Get the position, velocity and acceleration trajectories from the B-spline, based on the control points
    q_start = q_pos_trajs[:, 0, :]
    q_goal = q_pos_trajs[:, -1, :]
    q_trajs_d = planning_task.parametric_trajectory.get_q_trajectory(
        q_control_points, q_start, q_goal, get_type=("pos", "vel", "acc")
    )
    q_trajs_pos = q_trajs_d["pos"]
    q_trajs_vel = q_trajs_d["vel"]
    q_trajs_acc = q_trajs_d["acc"]

    # -------------------------------- Visualize ---------------------------------
    fig, axs = planning_task.plot_joint_space_trajectories(
        q_pos_trajs=q_trajs_pos,
        q_vel_trajs=q_trajs_vel,
        q_acc_trajs=q_trajs_acc,
        # control_points=control_points_concat,
        set_q_pos_limits=False,
        set_q_vel_limits=False,
        set_q_acc_limits=False,
    )
    fig.savefig(os.path.join(data_dir, f"figures/joint_space_trajectories.png"), bbox_inches="tight")

    # plot histogram of costs
    # costs_position = q_trajs_pos.pow(2).sum(dim=-1).sum(dim=-1)
    # fig, axs = plt.subplots(1, 1, squeeze=False)
    # axs[0, 0].hist(to_numpy(costs_position), bins=10)
    # seaborn.kdeplot(data=to_numpy(costs_position), ax=axs[0, 0].twinx())
    # axs[0, 0].set_title('Costs position histogram')
    # fig.savefig(os.path.join(DATA_DIR, f'figures/costs_position.png'), bbox_inches='tight')
    #
    # costs_velocity = q_trajs_vel.pow(2).sum(dim=-1).sum(dim=-1)
    # fig, axs = plt.subplots(1, 1, squeeze=False)
    # axs[0, 0].hist(to_numpy(costs_velocity), bins=10)
    # seaborn.kdeplot(data=to_numpy(costs_velocity), ax=axs[0, 0].twinx())
    # axs[0, 0].set_title('Costs velocity histogram')
    # fig.savefig(os.path.join(DATA_DIR, f'figures/costs_velocity.png'), bbox_inches='tight')
    #
    # costs_acceleration = q_trajs_acc.pow(2).sum(dim=-1).sum(dim=-1)
    # fig, axs = plt.subplots(1, 1, squeeze=False)
    # axs[0, 0].hist(to_numpy(costs_acceleration), bins=10)
    # seaborn.kdeplot(data=to_numpy(costs_acceleration), ax=axs[0, 0].twinx())
    # axs[0, 0].set_title('Costs acceleration histogram')
    # fig.savefig(os.path.join(DATA_DIR, f'figures/costs_acceleration.png'), bbox_inches='tight')

    ########################
    # Visualize in Pybullet
    generate_data = GenerateDataOMPL(
        args["env_id"],
        args["robot_id"],
        min_distance_robot_env=args["min_distance_robot_env"],
        pybullet_mode="DIRECT",
        tensor_args=tensor_args,
        debug=True,
    )

    path = to_numpy(q_trajs_pos[0])  # select only the first trajectory (pybullet does not allow parallelization)
    generate_data.pbompl_interface.execute(path, sleep_time=5.0 / len(path))

    ########################
    # Visualize in Isaac Gym
    # POSITION CONTROL
    # add initial positions for better visualization
    n_pre_steps = 10
    n_post_steps = 10

    if isaac_gym_render_all_trajectories:
        assert q_trajs_pos.shape[1] == 1
        q_pos_trajs_isaac = q_trajs_pos.squeeze()
    else:
        q_pos_trajs_isaac = q_trajs_pos

    q_pos_trajs_isaac = q_pos_trajs_isaac.movedim(1, 0)

    motion_planning_isaac_env = MotionPlanningIsaacGymEnv(
        env,
        robot,
        asset_root=get_robot_path().as_posix(),
        robot_asset_file=robot.robot_urdf_file.replace(get_robot_path().as_posix() + "/", ""),
        num_envs=q_pos_trajs_isaac.shape[1],
        all_robots_in_one_env=True,
        show_viewer=True,
        sync_viewer_with_real_time=False,
        viewer_time_between_steps=parametric_trajectory.phase_time.trajectory_duration / q_pos_trajs_isaac.shape[0],
        render_camera_global=True,
        render_camera_global_append_to_recorder=True,
        color_robots=False,
        # draw_goal_configuration=True if not args['sample_joint_position_goals_with_same_ee_pose'] else False,
        draw_goal_configuration=False,
        draw_collision_spheres=False,
        draw_contact_forces=False,
        draw_end_effector_frame=False,
        draw_end_effector_path=True,
        draw_ee_pose_goal=None,
        camera_global_from_top=True if env.dim == 2 else False,
        # add_ground_plane=False if env.dim == 2 else True,
        add_ground_plane=False,
    )

    motion_planning_controller = MotionPlanningControllerIsaacGym(motion_planning_isaac_env)
    isaac_statistics = motion_planning_controller.execute_trajectories(
        q_pos_trajs_isaac,
        q_pos_starts=q_pos_trajs_isaac[0],
        q_pos_goal=q_pos_trajs_isaac[-1][0],
        n_pre_steps=n_pre_steps,
        n_post_steps=n_post_steps,
        make_video=True,
        video_path=os.path.join(data_dir, f"figures/isaac-planning.mp4"),
        make_gif=False,
        save_step_images=True,
    )

    print("-----------------")
    print(f"isaac_statistics:")
    pprint(isaac_statistics)
    print("-----------------")


if __name__ == "__main__":
    visualize()
