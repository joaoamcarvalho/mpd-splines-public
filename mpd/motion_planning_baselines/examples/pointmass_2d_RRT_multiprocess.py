import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from mp_baselines.planners.multi_sample_based_planner import MultiSampleBasedPlanner
from mp_baselines.planners.rrt_connect import RRTConnect
from mp_baselines.planners.rrt_star import InfRRTStar
from torch_robotics.environments import EnvNarrowPassageDense2D
from torch_robotics.robots.robot_point_mass import RobotPointMass2D
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
    # tensor_args = {'device': 'cpu', 'dtype': torch.float32}

    # ---------------------------- Environment, Robot, PlanningTask ---------------------------------
    # env = EnvDense2D(
    #     precompute_sdf_obj_fixed=True,
    #     sdf_cell_size=0.005,
    #     tensor_args=tensor_args
    # )

    # env = EnvDense2DExtraObjects(
    #     precompute_sdf_obj_fixed=True,
    #     sdf_cell_size=0.005,
    #     tensor_args=tensor_args
    # )

    env = EnvNarrowPassageDense2D(precompute_sdf_obj_fixed=True, sdf_cell_size=0.005, tensor_args=tensor_args)

    robot = RobotPointMass2D(tensor_args=tensor_args)

    task = PlanningTask(
        env=env,
        robot=robot,
        # ws_limits=torch.tensor([[-0.85, -0.85], [0.95, 0.95]], **tensor_args),  # workspace limits
        # use_occupancy_map=True,  # whether to create and evaluate collisions on an occupancy map
        use_occupancy_map=False,
        cell_size=0.001,
        obstacle_cutoff_margin=0.005,
        tensor_args=tensor_args,
    )

    # -------------------------------- Planner ---------------------------------
    start_state_pos = torch.tensor([-0.9, -0.9], **tensor_args)
    # goal_state = torch.tensor([0.25, 0.9], **tensor_args)
    goal_state_pos = torch.tensor([0.9, 0.0], **tensor_args)

    n_iters = 30000
    step_size = 0.01
    n_radius = 0.1
    max_time = 60.0

    n_trajectories = 8  # use the cpu and increase to 100 to see the benefit of multiprocessing

    if planner == "rrt-connect":
        rrt_connect_params = dict(
            task=task,
            n_iters=n_iters,
            start_state_pos=start_state_pos,
            step_size=step_size,
            n_radius=n_radius,
            max_time=max_time,
            goal_state_pos=goal_state_pos,
            tensor_args=tensor_args,
        )
        planner = RRTConnect(**rrt_connect_params)
    elif planner == "rrt-star":
        max_best_cost_iters = 1000
        cost_eps = 1e-2
        n_knn = 10
        goal_prob = 0.1

        rrt_star_params = dict(
            task=task,
            n_iters=n_iters,
            max_best_cost_iters=max_best_cost_iters,
            cost_eps=cost_eps,
            start_state_pos=start_state_pos,
            step_size=step_size,
            n_radius=n_radius,
            n_knn=n_knn,
            max_time=max_time,
            goal_prob=goal_prob,
            goal_state_pos=goal_state_pos,
            tensor_args=tensor_args,
        )
        planner = InfRRTStar(**rrt_star_params)
    else:
        raise NotImplementedError

    # Optimize in parallel
    with TimerCUDA() as t:
        sample_based_planner = MultiSampleBasedPlanner(
            planner,
            n_trajectories=n_trajectories,
            # max_processes=8,
            optimize_sequentially=False,
        )
        trajs_l = sample_based_planner.optimize()
    print(f"Optimization time MultiProcessing: {t.elapsed:.3f} sec")

    # Optimize sequentially
    with TimerCUDA() as t:
        sample_based_planner = MultiSampleBasedPlanner(
            planner, n_trajectories=n_trajectories, optimize_sequentially=True
        )
        trajs_l = sample_based_planner.optimize()
    print(f"Optimization time ForLoop: {t.elapsed:.3f} sec")

    # -------------------------------- Visualize ---------------------------------

    base_file_name = Path(os.path.basename(__file__)).stem

    fig, ax = None, None
    for traj in trajs_l:
        if traj is None:
            continue
        traj_pos = task.get_position(traj).unsqueeze(0)  # add batch dimension for interface
        fig, ax = task.render_robot_trajectories(
            fig=fig, ax=ax, q_pos_trajs=traj_pos, start_state=start_state_pos, goal_state=goal_state_pos
        )

    plt.show()
