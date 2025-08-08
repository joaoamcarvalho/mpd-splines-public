import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

from mp_baselines.planners.rrt_connect import RRTConnect
from mp_baselines.planners.rrt_star import InfRRTStar
from torch_robotics.environments import EnvTableShelf
from torch_robotics.robots.robot_panda import RobotPanda
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device

allow_ops_in_compiled_graph()


if __name__ == "__main__":
    planner = "rrt-connect"
    # parametric_trajectory = 'rrt-star'

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
    q_free = task.random_coll_free_q_pos(n_samples=2)
    start_state = q_free[0]
    goal_state = q_free[1]

    # start_state = torch.tensor([-2.2248, -0.6046,  1.7909, -1.5844, -0.4575,  3.6484, -1.4562], **tensor_args)
    # goal_state = torch.tensor([0.1927,  1.2406, -0.4233, -1.6301, -2.7528,  2.5637, -0.7582], **tensor_args)

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
        n_iters = 30000
        step_size = torch.pi / 80
        n_radius = torch.pi / 4
        max_time = 120

        max_best_cost_iters = 2000
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

    print(f"----------------STATISTICS----------------")
    print(f"percentage free trajs: {task.compute_fraction_valid_trajs(pos_trajs_iters) * 100:.2f}")
    print(f"percentage collision intensity {task.compute_collision_intensity_trajs(pos_trajs_iters) * 100:.2f}")
    print(f"success {task.compute_success_valid_trajs(pos_trajs_iters)}")

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
        render_planner=False,
    )

    task.animate_robot_trajectories(
        q_pos_trajs=pos_trajs_iters,
        q_pos_start=start_state,
        q_pos_goal=goal_state,
        video_filepath=f"{base_file_name}-robot-traj.mp4",
        # n_frames=max((2, pos_trajs_iters[-1].shape[1]//10)),
        n_frames=pos_trajs_iters.shape[1],
        anim_time=5,
    )

    plt.show()
