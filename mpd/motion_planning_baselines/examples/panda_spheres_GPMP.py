from future.moves import pickle

from torch_robotics.environments import GraspedObjectBox, EnvSpheres3DExtraObjectsV00

import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

from mp_baselines.planners.gpmp2 import GPMP2
from torch_robotics.environments import EnvSpheres3D
from torch_robotics.robots.robot_panda import RobotPanda
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device

allow_ops_in_compiled_graph()


if __name__ == "__main__":
    base_file_name = Path(os.path.basename(__file__)).stem

    seed = 1111
    fix_random_seed(seed)

    device = get_torch_device()
    tensor_args = {"device": device, "dtype": torch.float32}

    # ---------------------------- Environment, Robot, PlanningTask ---------------------------------
    env = EnvSpheres3D(precompute_sdf_obj_fixed=True, sdf_cell_size=0.01, tensor_args=tensor_args)

    robot = RobotPanda(
        use_object_collision_spheres=True,
        # grasped_object=GraspedObjectPandaBox(tensor_args=tensor_args),
        tensor_args=tensor_args,
    )

    task = PlanningTask(
        env=env,
        robot=robot,
        ws_limits=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], **tensor_args),  # workspace limits
        obstacle_cutoff_margin=0.05,
        tensor_args=tensor_args,
    )

    # -------------------------------- Planner ---------------------------------
    for _ in range(100):
        q_free = task.random_coll_free_q_pos(n_samples=2)
        start_state = q_free[0]
        goal_state = q_free[1]

        # check if the EE positions are "enough" far apart
        start_state_ee_pos = robot.get_EE_position(start_state).squeeze()
        goal_state_ee_pos = robot.get_EE_position(goal_state).squeeze()

        if torch.linalg.norm(start_state_ee_pos - goal_state_ee_pos) > 0.5:
            break

    # start_state = torch.tensor([-2.6, 0.05, -1.2, -2.15,  1.33,  3.7, -1.7698],
    #    device='cuda:0')
    # goal_state = torch.tensor([ 0.9791, -0.2869,  2.0436, -0.4489, -0.2500,  1.6288,  2.0535],
    #    device='cuda:0')

    # start_state = torch.tensor([1.0403,  0.0493,  0.0251, -1.2673,  1.6676,  3.3611, -1.5428], **tensor_args)
    # goal_state = torch.tensor([1.1142,  1.7289, -0.1771, -0.9284,  2.7171,  1.2497,  1.7724], **tensor_args)

    print(start_state)
    print(goal_state)

    # Construct parametric_trajectory
    duration = 5  # sec
    n_support_points = 128
    dt = duration / n_support_points

    num_particles_per_goal = 10

    default_params_env = env.get_gpmp2_params(robot=robot)

    planner_params = dict(
        **default_params_env,
        robot=robot,
        n_dof=robot.q_dim,
        n_support_points=n_support_points,
        num_particles_per_goal=num_particles_per_goal,
        dt=dt,
        start_state=start_state,
        multi_goal_states=goal_state.unsqueeze(0),  # add batch dim for interface,
        collision_fields=task.get_all_collision_fields(),
        tensor_args=tensor_args,
    )
    planner = GPMP2(**planner_params)

    # Optimize
    opt_iters = default_params_env["opt_iters"]
    trajs_0 = planner.get_traj()
    trajs_iters = []
    trajs_iters.append(trajs_0)
    costs_previous = None
    with TimerCUDA() as t:
        for i in range(opt_iters):
            print(f"Iteration: {i}")
            trajs = planner.optimize(opt_iters=1, debug=True)
            trajs_iters.append(trajs)

            costs = planner.costs
            if i == 0:
                costs_previous = costs
                continue

            if torch.all(torch.abs((costs - costs_previous) / costs) < 0.1):
                break

            costs_previous = costs.clone()

    trajs_iters = torch.stack(trajs_iters)
    print(f"Optimization time: {t.elapsed:.3f} sec, per iteration: {t.elapsed/len(trajs_iters):.3f}")
    torch.cuda.empty_cache()

    # save trajectories
    trajs_iters_coll, trajs_iters_free = task.get_trajs_unvalid_and_valid(trajs_iters[-1])
    results_data_dict = {
        "duration": duration,
        "n_support_points": n_support_points,
        "dt": dt,
        "trajs_iters_coll": trajs_iters_coll.unsqueeze(0) if trajs_iters_coll is not None else None,
        "trajs_iters_free": trajs_iters_free.unsqueeze(0) if trajs_iters_free is not None else None,
    }

    with open(
        os.path.join("../../../deps/experiment_launcher/examples/", f"{base_file_name}-results_data_dict.pickle"), "wb"
    ) as handle:
        pickle.dump(results_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # -------------------------------- Visualize ---------------------------------
    print(f"----------------STATISTICS----------------")
    print(f"percentage free trajs: {task.compute_fraction_valid_trajs(trajs_iters[-1]) * 100:.2f}")
    print(f"percentage collision intensity {task.compute_collision_intensity_trajs(trajs_iters[-1])*100:.2f}")
    print(f"success {task.compute_success_valid_trajs(trajs_iters[-1])}")

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
        q_pos_trajs=pos_trajs_iters[-1, 0][None, ...][:, ::20, :],
        # trajs=pos_trajs_iters[-1, 0][None, ...],
        start_state=start_state,
        goal_state=goal_state,
        render_planner=False,
        draw_links_spheres=False,
    )

    task.animate_robot_trajectories(
        q_pos_trajs=pos_trajs_iters[-1, 0][None, ...],
        q_pos_start=start_state,
        q_pos_goal=goal_state,
        plot_x_trajs=False,
        draw_links_spheres=False,
        video_filepath=f"{base_file_name}-robot-traj.mp4",
        # n_frames=max((2, pos_trajs_iters[-1].shape[1]//10)),
        n_frames=pos_trajs_iters[-1].shape[1],
        anim_time=duration,
    )

    plt.show()
