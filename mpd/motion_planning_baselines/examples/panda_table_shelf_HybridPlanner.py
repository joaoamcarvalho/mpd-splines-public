import os
from pathlib import Path

import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
from matplotlib import pyplot as plt

from mp_baselines.planners.gpmp2 import GPMP2
from mp_baselines.planners.hybrid_planner import HybridPlanner
from mp_baselines.planners.multi_sample_based_planner import MultiSampleBasedPlanner
from mp_baselines.planners.rrt_connect import RRTConnect
from torch_robotics.environments import EnvTableShelf
from torch_robotics.robots.robot_panda import RobotPanda
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device

allow_ops_in_compiled_graph()


if __name__ == "__main__":
    seed = 2
    fix_random_seed(seed)

    device = get_torch_device()
    tensor_args = {"device": device, "dtype": torch.float32}

    # ---------------------------- Environment, Robot, PlanningTask ---------------------------------
    env = EnvTableShelf(precompute_sdf_obj_fixed=True, sdf_cell_size=0.01, tensor_args=tensor_args)

    robot = RobotPanda(
        # grasped_object=GraspedObjectPandaBox(tensor_args=tensor_args),
        tensor_args=tensor_args
    )

    task = PlanningTask(
        env=env,
        robot=robot,
        ws_limits=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], **tensor_args),  # workspace limits
        obstacle_cutoff_margin=0.03,
        tensor_args=tensor_args,
    )

    # -------------------------------- Planner ---------------------------------
    # q_free = tasks.random_coll_free_q(n_samples=2)
    # start_state = q_free[0]
    # goal_state = q_free[1]
    #
    for _ in range(100):
        q_free = task.random_coll_free_q_pos(n_samples=2)
        start_state = q_free[0]
        goal_state = q_free[1]

        # check if the EE positions are "enough" far apart
        start_state_ee_pos = robot.get_EE_position(start_state).squeeze()
        goal_state_ee_pos = robot.get_EE_position(goal_state).squeeze()

        if torch.linalg.norm(start_state - goal_state) > 0.5:
            break

    # start_state = torch.tensor([-0.7760, -1.0717, -2.4756, -1.4973, -2.2995, 0.7608, -1.3377],
    #        device='cuda:0')
    # goal_state = torch.tensor([-0.5312, -1.3097, 2.4938, -1.9871, 1.3979, 1.8733, -2.1781],
    #        device='cuda:0')

    # start_state = torch.tensor([ 1.9413, -0.0090,  2.3629, -0.8916,  0.2496,  3.5482, -0.7393], device='cuda:0')
    # goal_state = torch.tensor([-2.6686, -0.1020, -0.2527, -2.7064,  1.0567,  1.2865,  2.2158], device='cuda:0')

    print(start_state)
    print(goal_state)

    n_trajectories = 5

    ############### Sample-based parametric_trajectory
    rrt_connect_default_params_env = env.get_rrt_connect_params()

    rrt_connect_params = dict(
        **rrt_connect_default_params_env,
        task=task,
        start_state=start_state,
        goal_state=goal_state,
        tensor_args=tensor_args,
    )
    sample_based_planner_base = RRTConnect(**rrt_connect_params)
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
    planner_params["opt_iters"] = 250
    opt_based_planner = GPMP2(**planner_params)

    ############### Hybrid parametric_trajectory
    opt_iters = planner_params["opt_iters"]
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
        n_frames=max((2, opt_iters // 10)),
        anim_time=5,
    )

    task.render_robot_trajectories(
        q_pos_trajs=pos_trajs_iters[-1, 0][None, ...],
        start_state=start_state,
        goal_state=goal_state,
        render_planner=False,
    )

    task.animate_robot_trajectories(
        q_pos_trajs=pos_trajs_iters[-1, 0][None, ...],
        q_pos_start=start_state,
        q_pos_goal=goal_state,
        plot_x_trajs=False,
        video_filepath=f"{base_file_name}-robot-traj.mp4",
        # n_frames=max((2, pos_trajs_iters[-1].shape[1]//10)),
        n_frames=pos_trajs_iters[-1].shape[1],
        anim_time=n_support_points * dt,
    )

    plt.show()
