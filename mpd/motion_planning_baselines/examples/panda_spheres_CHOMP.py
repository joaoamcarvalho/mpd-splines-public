import isaacgym
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

from mp_baselines.planners.chomp import CHOMP
from mp_baselines.planners.costs.cost_functions import CostComposite, CostCollision
from mpd.parametric_trajectory.trajectory_waypoints import ParametricTrajectoryWaypoints
from torch_robotics.environments import EnvSpheres3D
from torch_robotics.environments import GraspedObjectBox
from torch_robotics.robots.robot_panda import RobotPanda
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device

allow_ops_in_compiled_graph()


if __name__ == "__main__":
    base_file_name = Path(os.path.basename(__file__)).stem

    seed = 11110779
    fix_random_seed(seed)

    device = get_torch_device()
    # device = 'cpu'
    print(device)
    tensor_args = {"device": device, "dtype": torch.float32}

    # ---------------------------- Environment, Robot, PlanningTask ---------------------------------
    env = EnvSpheres3D(precompute_sdf_obj_fixed=True, sdf_cell_size=0.02, tensor_args=tensor_args)

    robot = RobotPanda(
        grasped_object=GraspedObjectBox(
            attached_to_frame=RobotPanda.link_name_ee, object_collision_margin=0.05, tensor_args=tensor_args
        ),
        gripper=True,
        tensor_args=tensor_args,
    )

    task = PlanningTask(
        parametric_trajectory=ParametricTrajectoryWaypoints(
            n_control_points=64, num_T_pts=128, trajectory_duration=5.0, tensor_args=tensor_args
        ),
        env=env,
        robot=robot,
        # ws_limits=torch.tensor([[-0.85, -0.85], [0.95, 0.95]], **tensor_args),  # workspace limits
        obstacle_buffer=0.005,
        tensor_args=tensor_args,
    )

    # -------------------------------- Planner ---------------------------------
    q_free = task.random_coll_free_q_pos(n_samples=2)
    start_state_pos = q_free[0]
    goal_state_pos = q_free[1]
    print(f"start_state_pos: {start_state_pos}")
    print(f"goal_state_pos: {goal_state_pos}")

    multi_goal_states = goal_state_pos.unsqueeze(0)

    duration = 5  # sec
    n_support_points = 64
    dt = duration / n_support_points

    # Construct cost function
    sigma_coll = 1e-3
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
    opt_iters = 50

    planner_params = dict(
        n_dof=robot.q_dim,
        n_support_points=n_support_points,
        num_particles_per_goal=num_particles_per_goal,
        opt_iters=1,  # Keep this 1 for visualization
        dt=dt,
        start_state=start_state_pos,
        cost=cost_composite,
        weight_prior_cost=1e-4,
        step_size=0.05,
        grad_clip=0.05,
        multi_goal_states=multi_goal_states,
        sigma_start_init=0.001,
        sigma_goal_init=0.001,
        sigma_gp_init=0.3,
        pos_only=False,
        tensor_args=tensor_args,
    )

    planner = CHOMP(**planner_params)

    # Optimize
    trajs_0 = planner.get_traj()
    trajs_iters = torch.empty((opt_iters + 1, *trajs_0.shape), **tensor_args)
    trajs_iters[0] = trajs_0
    with TimerCUDA() as t:
        for i in range(opt_iters):
            trajs = planner.optimize(debug=True)
            trajs_iters[i + 1] = trajs
    print(f"Optimization time: {t.elapsed:.3f} sec, per iteration: {t.elapsed/opt_iters:.3f}")

    # save trajectories
    trajs_iters_coll, trajs_iters_free = task.get_trajs_unvalid_and_valid(trajs_iters[-1])
    results_data_dict = {
        "duration": duration,
        "n_support_points": n_support_points,
        "dt": dt,
        "trajs_iters_coll": trajs_iters_coll.unsqueeze(0) if trajs_iters_coll is not None else None,
        "trajs_iters_free": trajs_iters_free.unsqueeze(0) if trajs_iters_free is not None else None,
    }

    with open(os.path.join(f"{base_file_name}-results_data_dict.pickle"), "wb") as handle:
        pickle.dump(results_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # -------------------------------- Visualize ---------------------------------
    print(f"----------------STATISTICS----------------")
    print(f"percentage free trajs: {task.compute_fraction_valid_trajs(trajs_iters[-1]) * 100:.2f}")
    print(f"percentage collision intensity {task.compute_collision_intensity_trajs(trajs_iters[-1])*100:.2f}")
    print(f"success {task.compute_success_valid_trajs(trajs_iters[-1])}")

    base_file_name = Path(os.path.basename(__file__)).stem

    pos_trajs_iters = task.get_position(trajs_iters)

    task.plot_joint_space_trajectories(
        q_pos_trajs=trajs_iters[-1],
        q_pos_start=start_state_pos,
        q_pos_goal=goal_state_pos,
        q_vel_start=torch.zeros_like(start_state_pos),
        q_vel_goal=torch.zeros_like(goal_state_pos),
    )

    task.animate_opt_iters_joint_space_state(
        q_pos_trajs=trajs_iters,
        pos_start_state=start_state_pos,
        pos_goal_state=goal_state_pos,
        vel_start_state=torch.zeros_like(start_state_pos),
        vel_goal_state=torch.zeros_like(goal_state_pos),
        video_filepath=f"{base_file_name}-joint-space-opt-iters.mp4",
        n_frames=max((2, opt_iters // 10)),
        anim_time=5,
    )

    task.render_robot_trajectories(
        q_pos_trajs=pos_trajs_iters[-1, 0][None, ...],
        start_state=start_state_pos,
        goal_state=goal_state_pos,
        render_planner=False,
    )

    task.animate_robot_trajectories(
        q_pos_trajs=pos_trajs_iters[-1, 0][None, ...],
        q_pos_start=start_state_pos,
        q_pos_goal=goal_state_pos,
        plot_x_trajs=False,
        video_filepath=f"{base_file_name}-robot-traj.mp4",
        # n_frames=max((2, pos_trajs_iters[-1].shape[1]//10)),
        n_frames=pos_trajs_iters[-1].shape[1],
        anim_time=n_support_points * dt,
    )

    plt.show()
