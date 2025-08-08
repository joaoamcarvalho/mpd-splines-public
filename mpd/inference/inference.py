import os
from math import ceil
from pathlib import Path

import numpy as np
import torch
from dotmap import DotMap
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from mpd.models import GaussianDiffusionModel, guide_gradient_steps, CVAEModel
from mpd.utils.loaders import load_params_from_yaml
from pb_ompl.pb_ompl import add_box, fit_bspline_to_path
from scripts.generate_data.generate_trajectories import GenerateDataOMPL
from mpd.inference.cost_guides import CostGuideManagerParametricTrajectory, NoCostException
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import (
    to_numpy,
    freeze_torch_model_params,
    to_torch,
    dict_to_device,
    DEFAULT_TENSOR_ARGS,
)
from torch_robotics.trajectory.metrics import compute_smoothness, compute_ee_pose_errors, compute_path_length


class EvaluationSamplesGenerator:
    """
    Get start and goal joint positions from the validation set, or randomly using the OMPL parametric_trajectory
    """

    def __init__(
        self,
        planning_task,
        train_subset,
        val_subset,
        selection_start_goal="training",  # training, validation
        grasped_object=None,
        tensor_args=DEFAULT_TENSOR_ARGS,
        debug=False,
        render_pybullet=False,
        min_distance_q_pos_start_goal=None,
        **kwargs,
    ):
        self.tensor_args = tensor_args

        self.selection_start_goal = selection_start_goal
        self.select_start_goal_from_file = None
        self.dataset_subset = None
        self.idxs_dataset_subset = None
        self.train_subset = train_subset
        self.val_subset = val_subset
        if selection_start_goal == "training":
            self.dataset_subset = train_subset
            self.idxs_dataset_subset = np.random.permutation(len(train_subset))
        elif selection_start_goal == "validation":
            self.dataset_subset = val_subset
            self.idxs_dataset_subset = np.random.permutation(len(val_subset))
        else:
            self.select_start_goal_from_file = load_params_from_yaml(selection_start_goal)

        self.min_distance_q_pos_start_goal = None
        if min_distance_q_pos_start_goal is not None:
            self.min_distance_q_pos_start_goal = min_distance_q_pos_start_goal

        # OMPL worker to generate random start and goal joint positions
        self.generate_data_ompl_worker = GenerateDataOMPL(
            None,
            None,
            env_tr=planning_task.env,
            robot_tr=planning_task.robot,
            gripper=True,
            grasped_object=grasped_object,
            min_distance_robot_env=planning_task.min_distance_robot_env,
            tensor_args=tensor_args,
            pybullet_mode="GUI" if debug or render_pybullet else "DIRECT",
            debug=debug or render_pybullet,
        )

        self.ee_markers_ids = []

    def get_data_sample(self, idx, **kwargs):
        # -----------------------------------------------
        # Get the start and goal states
        # If extra objects are used, since they are not part of the original environment,
        # the start goal states from the training or validation sets might be in collision.
        # We just reject those samples and get new ones.
        if self.select_start_goal_from_file:
            idx = idx % len(self.select_start_goal_from_file)
            q_pos_start = to_torch(self.select_start_goal_from_file[idx]["q_pos_start"], **self.tensor_args)
            q_pos_goal = to_torch(self.select_start_goal_from_file[idx]["q_pos_goal"], **self.tensor_args)
            ee_pose_goal_flat = self.select_start_goal_from_file[idx]["ee_pose_goal"]
            ee_pose_goal = to_torch(ee_pose_goal_flat, **self.tensor_args).view(3, 4)
        else:
            idx = self.idxs_dataset_subset[idx % len(self.idxs_dataset_subset)]
            input_data_one_sample = self.dataset_subset[idx]
            q_pos_start = input_data_one_sample[self.dataset_subset.dataset.field_key_q_start]
            q_pos_goal = input_data_one_sample[self.dataset_subset.dataset.field_key_q_goal]
            ee_pose_goal = input_data_one_sample[self.dataset_subset.dataset.field_key_context_ee_goal_pose]

        if not self.generate_data_ompl_worker.pbompl_interface.is_state_valid(to_numpy(q_pos_start)):
            print("Start state is in collision. Getting new sample...")
            return self.get_data_sample(idx + 1)

        if not self.generate_data_ompl_worker.pbompl_interface.is_state_valid(to_numpy(q_pos_goal)):
            print("Goal state is in collision. Getting new sample...")
            return self.get_data_sample(idx + 1)

        q_pos_start = to_torch(q_pos_start, **self.tensor_args)
        q_pos_goal = to_torch(q_pos_goal, **self.tensor_args)

        if torch.linalg.norm(q_pos_goal - q_pos_start) < self.min_distance_q_pos_start_goal:
            print("Start and goal states are too close. Getting new sample...")
            return self.get_data_sample(idx + 1)

        ee_pose_goal = to_torch(ee_pose_goal, **self.tensor_args)

        return q_pos_start, q_pos_goal, ee_pose_goal

    def add_start_goal_marker(self, q_pos_start, q_pos_goal=None, ee_pose_goal=None, **kwargs):
        # remove markers first
        if self.ee_markers_ids:
            for marker_id in self.ee_markers_ids:
                self.generate_data_ompl_worker.pybullet_client.removeBody(marker_id)
            self.ee_markers_ids = []

        # adds a box to the pybullet environment to visualize the start and goal states
        ee_pose_start_np = self.generate_data_ompl_worker.pbompl_interface.get_ee_pose(to_numpy(q_pos_start))
        box_id = add_box(
            self.generate_data_ompl_worker.pybullet_client,
            ee_pose_start_np[0],
            [0.02] * 3,
            orientation=ee_pose_start_np[1],
            color=(0.0, 0.0, 1.0, 1.0),
        )
        self.ee_markers_ids.append(box_id)

        if ee_pose_goal is not None:
            ee_pose_goal_np = to_numpy(ee_pose_goal)
            ee_pose_goal_np = (ee_pose_goal_np[:3, -1], Rotation.from_matrix(ee_pose_goal_np[:3, :3]).as_quat())
        elif q_pos_goal is not None:
            ee_pose_goal_np = self.generate_data_ompl_worker.pbompl_interface.get_ee_pose(to_numpy(q_pos_goal))
        else:
            return
        box_id = add_box(
            self.generate_data_ompl_worker.pybullet_client,
            ee_pose_goal_np[0],
            [0.02] * 3,
            orientation=ee_pose_goal_np[1],
            color=(1.0, 0.0, 0.0, 1.0),
        )
        self.ee_markers_ids.append(box_id)


def render_results(
    args_inference,
    planning_task,
    q_pos_start,
    q_pos_goal,
    results_single_plan,
    idx,
    results_dir,
    render_joint_space_time_iters=False,
    render_joint_space_env_iters=False,
    render_planning_env_robot_opt_iters=False,
    render_planning_env_robot_trajectories=False,
    debug=False,
    **kwargs,
):
    base_file_name = Path(os.path.basename(__file__)).stem

    if results_single_plan.q_trajs_pos_best is not None:
        q_pos_traj_best = results_single_plan.q_trajs_pos_best
        q_vel_traj_best = results_single_plan.q_trajs_vel_best
        q_acc_traj_best = results_single_plan.q_trajs_acc_best
    else:
        q_pos_traj_best = None
        q_vel_traj_best = None
        q_acc_traj_best = None

    if render_joint_space_time_iters:
        planning_task.animate_opt_iters_joint_space_state(
            q_pos_trajs=results_single_plan.q_trajs_pos_iters,
            q_vel_trajs=results_single_plan.q_trajs_vel_iters,
            q_acc_trajs=results_single_plan.q_trajs_acc_iters,
            pos_start_state=q_pos_start,
            pos_goal_state=q_pos_goal,
            vel_start_state=torch.zeros_like(q_pos_start),
            vel_goal_state=torch.zeros_like(q_pos_goal),
            acc_start_state=torch.zeros_like(q_pos_start),
            acc_goal_state=torch.zeros_like(q_pos_goal),
            q_pos_traj_best=q_pos_traj_best,
            q_vel_traj_best=q_vel_traj_best,
            q_acc_traj_best=q_acc_traj_best,
            video_filepath=os.path.join(results_dir, f"{base_file_name}-joint_space-time-opt-iters-{idx:03d}.mp4"),
            n_frames=max((2, len(results_single_plan.q_trajs_pos_iters))),
            anim_time=args_inference.trajectory_duration,
            set_joint_limits=True,
            set_joint_vel_limits=True,
            set_joint_acc_limits=True,
            filter_joint_limits_vel_acc=True,
        )

        # reconstructed control points and trajectories at each diffusion iteration
        if results_single_plan.q_trajs_pos_recon_iters is not None:
            planning_task.animate_opt_iters_joint_space_state(
                q_pos_trajs=results_single_plan.q_trajs_pos_recon_iters,
                q_vel_trajs=results_single_plan.q_trajs_vel_recon_iters,
                q_acc_trajs=results_single_plan.q_trajs_acc_recon_iters,
                pos_start_state=q_pos_start,
                pos_goal_state=q_pos_goal,
                vel_start_state=torch.zeros_like(q_pos_start),
                vel_goal_state=torch.zeros_like(q_pos_goal),
                acc_start_state=torch.zeros_like(q_pos_start),
                acc_goal_state=torch.zeros_like(q_pos_goal),
                q_pos_traj_best=None,
                video_filepath=os.path.join(
                    results_dir, f"{base_file_name}-joint_space-time-opt-iters-recon-{idx:03d}.mp4"
                ),
                n_frames=max((2, len(results_single_plan.q_trajs_pos_iters))),
                anim_time=args_inference.trajectory_duration,
                set_joint_limits=True,
                set_joint_vel_limits=True,
                set_joint_acc_limits=True,
                filter_joint_limits_vel_acc=True,
            )

    if render_joint_space_env_iters:
        # visualize trajectories in the joint space
        planning_task.animate_opt_iters_joint_space_env(
            trajs_pos=results_single_plan.q_trajs_pos_iters,
            start_state=q_pos_start,
            goal_state=q_pos_goal,
            traj_pos_best=results_single_plan.q_trajs_pos_best,
            control_points=results_single_plan.control_points_iters,
            video_filepath=os.path.join(results_dir, f"{base_file_name}-joint_space-env-opt-iters-{idx:03d}.mp4"),
            n_frames=max((2, len(results_single_plan.q_trajs_pos_iters))),
            anim_time=args_inference.trajectory_duration,
            filter_joint_limits_vel_acc=True,
        )

    if render_planning_env_robot_opt_iters:
        # visualize in the planning environment
        planning_task.animate_opt_iters_robots(
            trajs_pos=results_single_plan.q_trajs_pos_iters,
            start_state=q_pos_start,
            goal_state=q_pos_goal,
            traj_pos_best=results_single_plan.q_trajs_pos_best,
            control_points=results_single_plan.control_points_iters,
            video_filepath=os.path.join(results_dir, f"{base_file_name}-robot-env-opt-iters-{idx:03d}.mp4"),
            n_frames=max((2, len(results_single_plan.q_trajs_pos_iters))),
            anim_time=args_inference.trajectory_duration,
            filter_joint_limits_vel_acc=True,
        )

        # reconstructed control points and trajectories at each diffusion iteration
        if results_single_plan.q_trajs_pos_recon_iters is not None:
            planning_task.animate_opt_iters_robots(
                trajs_pos=results_single_plan.q_trajs_pos_recon_iters,
                start_state=q_pos_start,
                goal_state=q_pos_goal,
                traj_pos_best=None,
                control_points=results_single_plan.control_points_recon_iters,
                video_filepath=os.path.join(results_dir, f"{base_file_name}-robot-env-opt-iters-recon-{idx:03d}.mp4"),
                n_frames=max((2, len(results_single_plan.q_trajs_pos_iters))),
                anim_time=args_inference.trajectory_duration,
                filter_joint_limits_vel_acc=True,
            )

    if render_planning_env_robot_trajectories:
        # visualize in the planning environment
        planning_task.animate_robot_trajectories(
            q_pos_trajs=results_single_plan.q_trajs_pos_iters[-1],
            q_pos_start=q_pos_start,
            q_pos_goal=q_pos_goal,
            plot_x_trajs=True,
            video_filepath=os.path.join(results_dir, f"{base_file_name}-robot-env-{idx:03d}.mp4"),
            n_frames=max((2, results_single_plan.q_trajs_pos_iters[-1].shape[1] // 10)),
            anim_time=args_inference.trajectory_duration,
            filter_joint_limits_vel_acc=True,
        )

    if debug:
        plt.show()


class GenerativeOptimizationPlanner:

    def __init__(
        self,
        planning_task,
        dataset,
        args_train,
        args_inference,
        tensor_args=DEFAULT_TENSOR_ARGS,
        sampling_based_planner_fn=None,
        debug=False,
        **kwargs,
    ):
        self.planning_task = planning_task
        self.dataset = dataset

        self.args_inference = args_inference

        self.tensor_args = tensor_args

        self.sampling_based_planner_fn = sampling_based_planner_fn

        self.debug = debug

        ################################################################################################################
        # Load the generative model
        # model_path = os.path.join(
        #     args_inference.model_dir, 'checkpoints',
        #     f'{"ema_" if args_train["use_ema"] else ""}model_current.pth'
        # )
        model_path = os.path.join(
            args_inference.model_dir, "checkpoints", f'{"ema_" if args_train["use_ema"] else ""}model_current.pth'
        )
        self.model = torch.load(model_path, map_location=tensor_args["device"])
        self.model.eval()
        freeze_torch_model_params(self.model)

        ################################################################################################################
        # Setup the generative model
        self.sample_fn_kwargs = {}
        if isinstance(self.model, GaussianDiffusionModel):
            diffusion_sampling_args = args_inference[args_inference.diffusion_sampling_method]
            if args_inference.diffusion_sampling_method == "ddpm":
                t_start_guide = ceil(
                    diffusion_sampling_args.t_start_guide_steps_fraction * self.model.n_diffusion_steps
                )
            elif args_inference.diffusion_sampling_method == "ddim":
                t_start_guide = ceil(
                    diffusion_sampling_args.t_start_guide_steps_fraction
                    * diffusion_sampling_args.ddim_sampling_timesteps
                )
            else:
                raise ValueError

            diffusion_sampling_args.update(
                method=args_inference.diffusion_sampling_method,
                t_start_guide=t_start_guide,
                n_diffusion_steps_without_noise=args_inference.n_diffusion_steps_without_noise,
                compute_costs_with_xrecon=args_inference.compute_costs_with_xrecon,
            )

            self.sample_fn_kwargs = diffusion_sampling_args
        elif isinstance(self.model, CVAEModel):
            pass
        else:
            raise NotImplementedError

        ################################################################################################################
        # Setup the costs and guided sampling
        self.cost_guide = None
        if args_inference.costs is not None:
            try:
                self.cost_guide = CostGuideManagerParametricTrajectory(
                    planning_task, dataset, args_inference, tensor_args, debug, **kwargs
                )
            except NoCostException:
                self.cost_guide = None

        ################################################################################################################
        # Warmup the model and guide costs
        self.warmup()

    def warmup(self, warmup_rounds=5, **kwargs):
        # cache the model for faster inference
        if self.debug:
            print(f"{'=' * 80}\nWarming up...\n{'=' * 80}")
        shape_x = (self.args_inference.n_trajectory_samples, *self.dataset.control_points_dim)
        for _ in range(warmup_rounds):
            self.model.warmup(shape_x, device=self.tensor_args["device"])
            if self.cost_guide is not None:
                self.cost_guide.warmup(shape_x)

    def plan_trajectory(
        self,
        q_pos_start,
        q_pos_goal,
        EE_pose_goal,
        n_trajectory_samples=None,
        results_ns: DotMap = None,
        debug=False,
        best_trajectory_selection="shortest_path_length",
        **kwargs,
    ):

        if results_ns is None:
            results_ns = DotMap()

        if n_trajectory_samples is None:
            n_trajectory_samples = self.args_inference.n_trajectory_samples

        # Prepare the input data and the context
        q_pos_start = to_torch(q_pos_start, **self.tensor_args)
        q_pos_goal = to_torch(q_pos_goal, **self.tensor_args)
        ee_pose_goal = to_torch(EE_pose_goal, **self.tensor_args)

        results_ns.update(
            q_pos_start=q_pos_start,
            q_pos_goal=q_pos_goal,
            ee_pose_goal=ee_pose_goal,
        )

        # Set the start and goal states
        self.planning_task.set_q_pos_start_goal(q_pos_start, q_pos_goal)
        self.planning_task.set_ee_pose_goal(ee_pose_goal)

        # Plan trajectories with the generative optimization planner
        # Get also the reconstructed control points
        input_data_one_sample = self.dataset.create_data_sample_normalized(
            q_pos_start,
            q_pos_goal,
            ee_pose_goal=ee_pose_goal,
        )
        input_data_one_sample = dict_to_device(input_data_one_sample, self.tensor_args["device"])
        hard_conds = input_data_one_sample["hard_conds"]
        context_d = self.dataset.build_context(input_data_one_sample)

        with TimerCUDA() as t_inference_total:
            control_points_recon_normalized_iters = None
            if "rrtconnect" in self.args_inference.planner_alg:
                with TimerCUDA() as t_generator:
                    assert self.sampling_based_planner_fn is not None, "sampling_based_planner_fn must be provided"
                    assert (
                        self.args_inference.n_trajectory_samples == 1
                    ), "n_trajectory_samples must be 1 for RRTConnect"
                    # Use the RRTConnect planner to get an initial trajectory
                    q_pos_start_np = to_numpy(q_pos_start, dtype=np.float64)
                    q_pos_goal_np = to_numpy(q_pos_goal, dtype=np.float64)
                    results_plan_d = self.sampling_based_planner_fn(1, q_pos_start_np, q_pos_goal_np)
                    bspline_params = fit_bspline_to_path(  # tck
                        results_plan_d[0]["sol_path"],
                        self.planning_task.parametric_trajectory.bspline.d,
                        self.planning_task.parametric_trajectory.bspline.n_pts,
                        self.planning_task.parametric_trajectory.zero_vel_at_start_and_goal,
                        self.planning_task.parametric_trajectory.zero_acc_at_start_and_goal,
                    )
                results_ns.update(
                    t_generator=t_generator.elapsed,
                )
                control_points_unnnormalized_np = bspline_params[1].T
                control_points_unnormalized_all = to_torch(control_points_unnnormalized_np, **self.tensor_args)[
                    None, None, ...
                ]
                control_points_unnormalized_iters = self.planning_task.parametric_trajectory.remove_control_points_fn(
                    control_points_unnormalized_all,
                )
                control_points_normalized_iters = self.dataset.normalize_control_points(
                    control_points_unnormalized_iters
                )

            elif "gp_prior" in self.args_inference.planner_alg:
                # Construct a GP trajectory prior between the start and goal states
                # If the EE goal pose is given, we use the q_pos_goal from the dataset, which can be understood as
                # doing inverse kinematics before planning
                ts = np.array([[0.0], [1.0]])
                xs = np.array([to_numpy(q_pos_start), to_numpy(q_pos_goal)])

                length_scale_bound_lower = np.linalg.norm(xs[0] - xs[1]) / 2
                print(f"length_scale_bound_lower: {length_scale_bound_lower}")
                kernel = 1 * RBF(
                    length_scale=length_scale_bound_lower, length_scale_bounds=(length_scale_bound_lower, 1e4)
                )
                gaussian_process = GaussianProcessRegressor(
                    kernel=kernel,
                    # optimizer=None,
                    n_restarts_optimizer=10,
                )
                gaussian_process.fit(ts, xs)
                print(gaussian_process.kernel_)

                ts_ = np.linspace(0, 1, self.dataset.n_learnable_control_points + 2)[:, None]
                xs_samples = gaussian_process.sample_y(
                    ts_, n_samples=self.args_inference.n_trajectory_samples, random_state=None
                )
                xs_samples = np.moveaxis(xs_samples, -1, 0)

                control_points_unnormalized_iters = to_torch(xs_samples[:, 1:-1, :], **self.tensor_args)[None, ...]
                control_points_normalized_iters = self.dataset.normalize_control_points(
                    control_points_unnormalized_iters
                )

            elif "knn_prior" in self.args_inference.planner_alg:
                # Get the K nearest neighbors from the dataset
                control_points_normalized = self.dataset.get_knn_control_points(
                    q_pos_start, q_pos_goal, EE_pose_goal, k=self.args_inference.n_trajectory_samples, normalized=True
                )
                control_points_normalized_iters = control_points_normalized[None, ...]

            else:
                control_points_normalized_iters = self.model.run_inference(
                    guide=self.cost_guide if self.args_inference.planner_alg == "mpd" else None,
                    context_d=context_d,
                    hard_conds=hard_conds,
                    n_samples=n_trajectory_samples,
                    horizon=self.dataset.n_learnable_control_points,
                    return_chain=True,
                    return_chain_x_recon=False,
                    results_ns=results_ns,
                    **self.sample_fn_kwargs,
                    debug=debug,
                )

            # run additional guide steps for the prior + guide planner
            if self.cost_guide is not None and (
                self.args_inference.planner_alg
                in [
                    "diffusion_prior_then_guide",
                    "gp_prior_then_guide",
                    "knn_prior_then_guide",
                    "cvae_prior_then_guide",
                    "rrtconnect_then_guide",
                ]
            ):
                with TimerCUDA() as t_guide:
                    # the same number of guide steps as used in MPD
                    sample_fn_kwargs_copy = self.sample_fn_kwargs.copy()
                    sample_fn_kwargs_copy.update(**self.args_inference[self.args_inference.planner_alg])
                    sample_fn_kwargs_copy.update(n_guide_steps=1)
                    control_points_normalized_iters_post = [control_points_normalized_iters[-1].detach().clone()]
                    for _ in range(self.args_inference[self.args_inference.planner_alg].n_guide_steps):
                        control_points_normalized_tmp = guide_gradient_steps(
                            control_points_normalized_iters_post[-1].detach().clone(),
                            hard_conds=hard_conds,
                            context_d=context_d,
                            guide=self.cost_guide,
                            **sample_fn_kwargs_copy,
                        )
                        control_points_normalized_iters_post.append(control_points_normalized_tmp)
                    control_points_normalized_iters = torch.cat(
                        [control_points_normalized_iters, torch.stack(control_points_normalized_iters_post)], dim=0
                    )
                results_ns.t_guide = t_guide.elapsed

            # run additional motion planning gradient steps
            if self.cost_guide is not None and self.args_inference.extra_mp_steps > 0:
                with TimerCUDA() as t_guide_extra_mp_steps:
                    sample_fn_kwargs_copy = self.sample_fn_kwargs.copy()
                    # use the same arguments are the diffusion prior + guide planner
                    sample_fn_kwargs_copy.update(**self.args_inference["diffusion_prior_then_guide"])
                    sample_fn_kwargs_copy.update(n_guide_steps=1)
                    control_points_normalized_iters_post = [control_points_normalized_iters[-1].detach().clone()]
                    # update the CostTaskSpaceCollisionObjects cost to use all the collision objects
                    self.cost_guide.use_all_collision_objects()
                    for _ in range(self.args_inference.extra_mp_steps):
                        control_points_normalized_tmp = guide_gradient_steps(
                            control_points_normalized_iters_post[-1].detach().clone(),
                            hard_conds=hard_conds,
                            context_d=context_d,
                            guide=self.cost_guide,
                            **sample_fn_kwargs_copy,
                        )
                        control_points_normalized_iters_post.append(control_points_normalized_tmp)
                    control_points_normalized_iters = torch.cat(
                        [control_points_normalized_iters, torch.stack(control_points_normalized_iters_post)], dim=0
                    )

                results_ns.t_guide += t_guide_extra_mp_steps.elapsed

        results_ns.t_inference_total = t_inference_total.elapsed

        # unnormalize control point samples from the models and get the trajectory from the control points
        control_points_iters = self.dataset.unnormalize_control_points(control_points_normalized_iters)
        q_trajs_pos_iters, q_trajs_vel_iters, q_trajs_acc_iters = self.compute_trajectories_from_control_points(
            q_pos_start, q_pos_goal, control_points_iters
        )
        if control_points_recon_normalized_iters is not None:
            control_points_recon_iters = self.dataset.unnormalize_control_points(control_points_recon_normalized_iters)
            q_trajs_pos_recon_iters, q_trajs_vel_recon_iters, q_trajs_acc_recon_iters = (
                self.compute_trajectories_from_control_points(q_pos_start, q_pos_goal, control_points_recon_iters)
            )
        else:
            control_points_recon_iters = None
            q_trajs_pos_recon_iters = None
            q_trajs_vel_recon_iters = None
            q_trajs_acc_recon_iters = None

        # Filter the valid trajectories
        control_points_iter_0 = control_points_iters[-1]
        q_trajs_pos_iter_0 = q_trajs_pos_iters[-1]
        q_trajs_vel_iter_0 = q_trajs_vel_iters[-1]
        q_trajs_acc_iter_0 = q_trajs_acc_iters[-1]
        q_trajs_iter_0 = torch.cat([q_trajs_pos_iter_0, q_trajs_vel_iter_0, q_trajs_acc_iter_0], dim=-1)
        _, _, q_trajs_final_valid, valid_idxs, _ = self.planning_task.get_trajs_unvalid_and_valid(
            q_trajs_iter_0,
            return_indices=True,
            filter_joint_limits_vel_acc=True,
        )
        if valid_idxs.ndim == 2:
            valid_idxs = valid_idxs.squeeze(1)

        control_points_valid = control_points_iter_0[valid_idxs]
        q_trajs_pos_valid = q_trajs_pos_iter_0[valid_idxs]
        q_trajs_vel_valid = q_trajs_vel_iter_0[valid_idxs]
        q_trajs_acc_valid = q_trajs_acc_iter_0[valid_idxs]

        # Get the "best" trajectory from all the valid ones
        if valid_idxs.numel() == 0:
            control_points_best = None
            q_trajs_pos_best = None
            q_trajs_vel_best = None
            q_trajs_acc_best = None
        else:
            best_trajectory_selection = self.args_inference.get("best_trajectory_selection", best_trajectory_selection)

            if self.dataset.context_ee_goal_pose:
                # Best = lowest EE pose cost
                # If the context is the EE pose goal, we the best trajectory is the one that is closest to the goal
                ee_pose_goal_achieved = self.planning_task.robot.get_EE_pose(q_trajs_pos_valid[..., -1, :])
                error_ee_pose_goal_position, error_ee_pose_goal_orientation = compute_ee_pose_errors(
                    ee_pose_goal, ee_pose_goal_achieved
                )
                ee_pose_goal_error_position_norm = torch.linalg.norm(error_ee_pose_goal_position, dim=-1)
                ee_pose_goal_error_orientation_norm = torch.rad2deg(
                    torch.linalg.norm(error_ee_pose_goal_orientation, dim=-1)
                )
                idx_min_cost = torch.argmin(ee_pose_goal_error_position_norm)
            else:
                if best_trajectory_selection == "lowest_weighted_cost":
                    # Best = lowest weighted cost
                    costs_valid, *_ = self.cost_guide(control_points_valid, return_cost=True)
                    idx_min_cost = torch.argmin(costs_valid)
                elif best_trajectory_selection == "lowest_smoothness_cost":
                    # Best = lowest smoothness cost
                    batch_smoothness = compute_smoothness(
                        q_trajs_pos_valid, self.planning_task.robot, trajs_acc=q_trajs_acc_valid
                    )
                    idx_min_cost = torch.argmin(batch_smoothness)
                elif best_trajectory_selection == "shortest_path_length":
                    # Best = lowest path
                    batch_path_length = compute_path_length(q_trajs_pos_valid, self.planning_task.robot)
                    idx_min_cost = torch.argmin(batch_path_length)
                else:
                    raise NotImplementedError

            control_points_best = control_points_valid[idx_min_cost]
            q_trajs_pos_best = q_trajs_pos_valid[idx_min_cost]
            q_trajs_vel_best = q_trajs_vel_valid[idx_min_cost]
            q_trajs_acc_best = q_trajs_acc_valid[idx_min_cost]

        results_ns.update(
            # control points and trajectories at each diffusion iteration
            control_points_iters=control_points_iters,
            q_trajs_pos_iters=q_trajs_pos_iters,
            q_trajs_vel_iters=q_trajs_vel_iters,
            q_trajs_acc_iters=q_trajs_acc_iters,
            # reconstructed control points and trajectories at each diffusion iteration
            control_points_recon_iters=control_points_recon_iters,
            q_trajs_pos_recon_iters=q_trajs_pos_recon_iters,
            q_trajs_vel_recon_iters=q_trajs_vel_recon_iters,
            q_trajs_acc_recon_iters=q_trajs_acc_recon_iters,
            # control points and trajectories at the last iteration
            control_points_iter_0=control_points_iter_0,
            q_trajs_pos_iter_0=q_trajs_pos_iter_0,
            q_trajs_vel_iter_0=q_trajs_vel_iter_0,
            q_trajs_acc_iter_0=q_trajs_acc_iter_0,
            # valid control points and trajectories
            control_points_valid=control_points_valid,
            q_trajs_pos_valid=q_trajs_pos_valid,
            q_trajs_vel_valid=q_trajs_vel_valid,
            q_trajs_acc_valid=q_trajs_acc_valid,
            # best control points and trajectories
            control_points_best=control_points_best,
            q_trajs_pos_best=q_trajs_pos_best,
            q_trajs_vel_best=q_trajs_vel_best,
            q_trajs_acc_best=q_trajs_acc_best,
            # trajectory time steps
            timesteps=self.planning_task.parametric_trajectory.get_timesteps(num=q_trajs_pos_iter_0.shape[1]),
        )
        return results_ns

    def compute_trajectories_from_control_points(self, q_pos_start, q_pos_goal, control_points, **kwargs):
        # Get the position, velocity and acceleration trajectories
        q_traj_d = self.planning_task.parametric_trajectory.get_q_trajectory(
            control_points, q_pos_start, q_pos_goal, get_type=("pos", "vel", "acc"), get_time_representation=True
        )
        q_trajs_pos_iters = q_traj_d["pos"]
        q_trajs_vel_iters = q_traj_d["vel"]
        q_trajs_acc_iters = q_traj_d["acc"]
        return q_trajs_pos_iters, q_trajs_vel_iters, q_trajs_acc_iters
