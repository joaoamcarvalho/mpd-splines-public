import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

from mp_baselines.planners.costs.cost_functions import CostGP, CostGoalPrior, CostComposite, CostCollision
from mp_baselines.planners.gpmp2 import GPMP2
from mp_baselines.planners.hybrid_planner import HybridPlanner
from mp_baselines.planners.multi_sample_based_planner import MultiSampleBasedPlanner
from mp_baselines.planners.rrt_connect import RRTConnect
from mp_baselines.planners.rrt_star import RRTStar
from torch_robotics.environments import EnvDense2D
from torch_robotics.environments import EnvGridCircles2D
from torch_robotics.environments import EnvSquare2D
from torch_robotics.robots.robot_point_mass import RobotPointMass2D
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device

allow_ops_in_compiled_graph()


if __name__ == "__main__":
    seed = 1
    fix_random_seed(seed)

    device = get_torch_device()
    tensor_args = {"device": device, "dtype": torch.float32}

    # ---------------------------- Environment, Robot, PlanningTask ---------------------------------
    env = EnvSquare2D(precompute_sdf_obj_fixed=True, sdf_cell_size=0.01, tensor_args=tensor_args)

    robot = RobotPointMass2D(tensor_args=tensor_args)

    task = PlanningTask(
        env=env,
        robot=robot,
        # ws_limits=torch.tensor([[-0.81, -0.81], [0.95, 0.95]], **tensor_args),  # workspace limits
        obstacle_buffer=0.005,
        tensor_args=tensor_args,
    )

    # -------------------------------- Planner ---------------------------------
    start_state = torch.tensor([-0.8, -0], **tensor_args)
    goal_state = torch.tensor([0.8, 0.0], **tensor_args)

    n_trajectories = 10

    ############### Sample-based parametric_trajectory
    rrt_connect_default_params_env = env.get_rrt_connect_params(robot=robot)

    rrt_connect_params = dict(
        **rrt_connect_default_params_env,
        task=task,
        start_state=start_state,
        goal_state=goal_state,
        tensor_args=tensor_args,
    )
    sample_based_planner_base = RRTConnect(**rrt_connect_params)
    # sample_based_planner_base = RRTStar(**rrt_connect_params)
    sample_based_planner = MultiSampleBasedPlanner(
        sample_based_planner_base, n_trajectories=n_trajectories, max_processes=8, optimize_sequentially=True
    )

    ############### Optimization-based parametric_trajectory
    n_support_points = 64
    dt = 0.04

    gpmp_default_params_env = env.get_gpmp2_params()

    # Construct parametric_trajectory
    planner_params = dict(
        **gpmp_default_params_env,
        robot=robot,
        n_dof=robot.q_dim,
        n_support_points=n_support_points,
        num_particles_per_goal=n_trajectories,
        dt=dt,
        start_state=start_state,
        multi_goal_states=goal_state.unsqueeze(0),  # add batch dim for interface,
        collision_fields=task.get_all_collision_fields(),
        tensor_args=tensor_args,
    )
    opt_based_planner = GPMP2(**planner_params)

    ############### Hybrid parametric_trajectory
    planner = HybridPlanner(sample_based_planner, opt_based_planner, tensor_args=tensor_args)

    trajs_iters = planner.optimize(debug=True, return_iterations=True)

    # -------------------------------- Visualize ---------------------------------
    print(f"----------------STATISTICS----------------")
    print(f"percentage free trajs: {task.compute_fraction_valid_trajs(trajs_iters[-1]) * 100:.2f}")
    print(f"percentage collision intensity {task.compute_collision_intensity_trajs(trajs_iters[-1])*100:.2f}")
    print(f"success {task.compute_success_valid_trajs(trajs_iters[-1])}")

    base_file_name = Path(os.path.basename(__file__)).stem

    pos_trajs_iters = task.get_position(trajs_iters)

    task.plot_joint_space_trajectories(
        q_pos_trajs=trajs_iters[-1],
        q_pos_start=start_state,
        q_pos_goal=goal_state,
        q_vel_start=torch.zeros_like(start_state),
        q_vel_goal=torch.zeros_like(goal_state),
    )

    task.animate_opt_iters_joint_space_state(
        q_pos_trajs=trajs_iters,
        pos_start_state=start_state,
        pos_goal_state=goal_state,
        vel_start_state=torch.zeros_like(start_state),
        vel_goal_state=torch.zeros_like(goal_state),
        video_filepath=f"{base_file_name}-joint-space-opt-iters.mp4",
        n_frames=max((2, gpmp_default_params_env["opt_iters"] // 10)),
        anim_time=5,
    )

    task.render_robot_trajectories(
        q_pos_trajs=pos_trajs_iters[-1],
        start_state=start_state,
        goal_state=goal_state,
        render_planner=False,
    )

    task.animate_robot_trajectories(
        q_pos_trajs=pos_trajs_iters[-1],
        q_pos_start=start_state,
        q_pos_goal=goal_state,
        plot_x_trajs=True,
        video_filepath=f"{base_file_name}-robot-traj.mp4",
        # n_frames=max((2, pos_trajs_iters[-1].shape[1]//10)),
        n_frames=pos_trajs_iters[-1].shape[1],
        anim_time=n_support_points * dt,
    )

    task.animate_opt_iters_robots(
        trajs_pos=pos_trajs_iters,
        start_state=start_state,
        goal_state=goal_state,
        video_filepath=f"{base_file_name}-traj-opt-iters.mp4",
        n_frames=max((2, gpmp_default_params_env["opt_iters"] // 10)),
        anim_time=5,
    )

    plt.show()
