import isaacgym
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

from mp_baselines.planners.chomp import CHOMP
from mp_baselines.planners.costs.cost_functions import CostCollision, CostComposite
from mpd.parametric_trajectory.trajectory_waypoints import ParametricTrajectoryWaypoints
from torch_robotics.environments import EnvSimple2D
from torch_robotics.robots.robot_point_mass import RobotPointMass2D
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device

allow_ops_in_compiled_graph()


if __name__ == "__main__":
    seed = 3
    fix_random_seed(seed)

    device = get_torch_device()
    tensor_args = {"device": device, "dtype": torch.float32}

    # ---------------------------- Environment, Robot, PlanningTask ---------------------------------
    # env = EnvDense2D(
    #     precompute_sdf_obj_fixed=True,
    #     sdf_cell_size=0.005,
    #     tensor_args=tensor_args
    # )

    env = EnvSimple2D(precompute_sdf_obj_fixed=True, sdf_cell_size=0.005, tensor_args=tensor_args)

    # env = EnvDense2DExtraObjects(
    #     precompute_sdf_obj_fixed=True,
    #     sdf_cell_size=0.005,
    #     tensor_args=tensor_args
    # )

    # env = EnvNarrowPassageDense2D(
    #     precompute_sdf_obj_fixed=True,
    #     sdf_cell_size=0.005,
    #     tensor_args=tensor_args
    # )

    robot = RobotPointMass2D(tensor_args=tensor_args)

    task = PlanningTask(
        parametric_trajectory=ParametricTrajectoryWaypoints(
            n_control_points=64, num_T_pts=128, trajectory_duration=3.0, tensor_args=tensor_args
        ),
        env=env,
        robot=robot,
        # ws_limits=torch.tensor([[-0.85, -0.85], [0.95, 0.95]], **tensor_args),  # workspace limits
        obstacle_buffer=0.005,
        tensor_args=tensor_args,
    )

    # -------------------------------- Planner ---------------------------------
    for _ in range(100):
        q_pos_free = task.random_coll_free_q_pos(n_samples=2)
        q_pos_start = q_pos_free[0]
        q_pos_goal = q_pos_free[1]

        if torch.linalg.norm(q_pos_start - q_pos_goal) > 1.0:
            break

    # start_state = torch.tensor([-0.2275, -0.0472], **tensor_args)
    # goal_state = torch.tensor([0.5302, 0.9507], **tensor_args)

    q_pos_goal_multi = q_pos_goal.unsqueeze(0)

    # Construct cost functions
    default_params_env = env.get_chomp_params(robot=robot)
    n_support_points = task.parametric_trajectory.num_T_pts
    dt = task.parametric_trajectory.trajectory_duration / default_params_env["n_support_points"]
    default_params_env.update(
        n_support_points=n_support_points,
        dt=dt,
    )

    cost_collisions = []
    weights_cost_l = []
    for collision_field in task.get_all_collision_fields():
        cost_collisions.append(
            CostCollision(robot, n_support_points, field=collision_field, sigma_coll=1.0, tensor_args=tensor_args)
        )
        weights_cost_l.append(10.0)

    cost_func_list = [*cost_collisions]
    cost_composite = CostComposite(
        robot, n_support_points, cost_func_list, weights_cost_l=weights_cost_l, tensor_args=tensor_args
    )

    num_particles_per_goal = 10
    opt_iters = 100

    planner_params = dict(
        **default_params_env,
        n_dof=robot.q_dim,
        num_particles_per_goal=num_particles_per_goal,
        start_state=q_pos_start,
        multi_goal_states=q_pos_goal.unsqueeze(0),  # add batch dim for interface,
        cost=cost_composite,
        tensor_args=tensor_args,
    )
    planner = CHOMP(**planner_params)

    # Optimize
    q_trajs_0 = planner.get_traj()
    q_trajs_iters = torch.empty((opt_iters + 1, *q_trajs_0.shape), **tensor_args)
    q_trajs_iters[0] = q_trajs_0
    with TimerCUDA() as t:
        for i in range(opt_iters):
            q_trajs = planner.optimize(debug=True)
            q_trajs_iters[i + 1] = q_trajs
    print(f"Optimization time: {t.elapsed:.3f} sec, per iteration: {t.elapsed/opt_iters:.3f}")

    # -------------------------------- Visualize ---------------------------------
    print(f"----------------STATISTICS----------------")
    print(f"percentage free trajs: {task.compute_fraction_valid_trajs(q_trajs_iters[-1]) * 100:.2f}")
    print(f"percentage collision intensity {task.compute_collision_intensity_trajs(q_trajs_iters[-1]) * 100:.2f}")
    print(f"success {task.compute_success_valid_trajs(q_trajs_iters[-1])}")

    base_file_name = Path(os.path.basename(__file__)).stem

    q_pos_trajs_iters = task.get_position(q_trajs_iters)
    q_vel_trajs_iters = task.get_velocity(q_trajs_iters)

    task.plot_joint_space_trajectories(
        q_pos_trajs=q_pos_trajs_iters[-1],
        q_vel_trajs=q_vel_trajs_iters[-1],
        q_pos_start=q_pos_start,
        q_pos_goal=q_pos_goal,
        q_vel_start=torch.zeros_like(q_pos_start),
        q_vel_goal=torch.zeros_like(q_pos_goal),
    )

    task.animate_opt_iters_joint_space_state(
        q_pos_trajs=q_pos_trajs_iters,
        q_vel_trajs=q_vel_trajs_iters,
        q_pos_start=q_pos_start,
        q_pos_goal=q_pos_goal,
        q_vel_start=torch.zeros_like(q_pos_start),
        q_vel_goal=torch.zeros_like(q_pos_goal),
        video_filepath=f"{base_file_name}-joint-space-opt-iters.mp4",
        n_frames=max((2, opt_iters // 10)),
        anim_time=5,
    )

    task.render_robot_trajectories(
        q_pos_trajs=q_pos_trajs_iters[-1],
        q_pos_start=q_pos_start,
        q_pos_goal=q_pos_goal,
        render_planner=False,
    )

    task.animate_robot_trajectories(
        q_pos_trajs=q_pos_trajs_iters[-1],
        q_pos_start=q_pos_start,
        q_pos_goal=q_pos_goal,
        plot_x_trajs=True,
        video_filepath=f"{base_file_name}-robot-traj.mp4",
        # n_frames=max((2, pos_trajs_iters[-1].shape[1]//10)),
        n_frames=q_pos_trajs_iters[-1].shape[1],
        anim_time=task.parametric_trajectory.trajectory_duration,
    )
    plt.show()

    task.animate_opt_iters_robots(
        trajs_pos=q_pos_trajs_iters,
        start_state=q_pos_start,
        goal_state=q_pos_goal,
        video_filepath=f"{base_file_name}-traj-opt-iters.mp4",
        n_frames=max((2, opt_iters // 10)),
        anim_time=5,
    )

    plt.show()
