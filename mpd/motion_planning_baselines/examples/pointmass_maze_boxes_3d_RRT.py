import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from mp_baselines.planners.rrt_connect import RRTConnect
from mp_baselines.planners.rrt_star import InfRRTStar
from torch_robotics.environments import EnvMazeBoxes3D
from torch_robotics.robots.robot_point_mass import RobotPointMass3D
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device

from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

allow_ops_in_compiled_graph()


if __name__ == "__main__":
    planner = "rrt-connect"
    # parametric_trajectory = 'rrt-star'

    seed = 0
    fix_random_seed(seed)

    device = get_torch_device()
    tensor_args = {"device": device, "dtype": torch.float32}

    # ---------------------------- Environment, Robot, PlanningTask ---------------------------------
    env = EnvMazeBoxes3D(precompute_sdf_obj_fixed=True, sdf_cell_size=0.01, tensor_args=tensor_args)

    robot = RobotPointMass3D(tensor_args=tensor_args)

    task = PlanningTask(
        env=env,
        robot=robot,
        # ws_limits=torch.tensor([[-1, -1, -1], [1, 1, 1]], **tensor_args),  # workspace limits
        obstacle_buffer=0.005,
        tensor_args=tensor_args,
    )

    # -------------------------------- Planner ---------------------------------
    start_state = torch.tensor([-0.8, -0.8, -0.8], **tensor_args)
    goal_state = torch.tensor([0.8, 0.8, 0.8], **tensor_args)

    n_iters = 30000
    step_size = 0.05
    n_radius = 0.3
    max_time = 60.0

    if planner == "rrt-connect":
        rrt_connect_default_params = env.get_rrt_connect_params()
        rrt_connect_params = dict(
            **rrt_connect_default_params,
            task=task,
            start_state=start_state,
            goal_state=goal_state,
            tensor_args=tensor_args,
        )
        planner = RRTConnect(**rrt_connect_params)
    elif planner == "rrt-star":
        max_best_cost_iters = 1000
        cost_eps = 1e-2
        n_knn = 10
        goal_prob = 0.2

        rrt_star_params = dict(
            task=task,
            n_iters=n_iters,
            max_best_cost_iters=max_best_cost_iters,
            cost_eps=cost_eps,
            start_state=start_state,
            step_size=step_size,
            n_radius=n_radius,
            n_knn=n_knn,
            max_time=max_time,
            goal_prob=goal_prob,
            goal_state=goal_state,
            tensor_args=tensor_args,
        )

        planner = InfRRTStar(**rrt_star_params)
    else:
        raise NotImplementedError

    # Optimize
    with TimerCUDA() as t:
        traj = planner.optimize(debug=True, refill_samples_buffer=True)
    print(f"Optimization time: {t.elapsed:.3f} sec")

    # -------------------------------- Visualize ---------------------------------

    base_file_name = Path(os.path.basename(__file__)).stem

    traj = traj.unsqueeze(0)  # batch dimension for interface

    pos_trajs_iters = task.get_position(traj)

    task.plot_joint_space_trajectories(
        q_pos_trajs=traj,
        q_pos_start=start_state,
        q_pos_goal=goal_state,
        q_vel_start=torch.zeros_like(start_state),
        q_vel_goal=torch.zeros_like(goal_state),
    )

    task.render_robot_trajectories(
        q_pos_trajs=pos_trajs_iters,
        start_state=start_state,
        goal_state=goal_state,
        render_planner=True,
    )

    task.animate_robot_trajectories(
        q_pos_trajs=pos_trajs_iters,
        q_pos_start=start_state,
        q_pos_goal=goal_state,
        plot_x_trajs=True,
        video_filepath=f"{base_file_name}-robot-traj.mp4",
        # n_frames=max((2, pos_trajs_iters[-1].shape[1]//10)),
        n_frames=pos_trajs_iters.shape[1],
        anim_time=5,
    )

    plt.show()
