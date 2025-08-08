import functools
import itertools
import sys
from abc import ABC
from functools import partial

import einops
import numpy as np
import torch
from matplotlib import pyplot as plt

from mpd.parametric_trajectory.trajectory_base import ParametricTrajectoryBase
from mpd.plotting.utils import remove_axes_labels_ticks
from torch_robotics.torch_planning_objectives.fields.distance_fields import (
    CollisionWorkspaceBoundariesDistanceField,
    CollisionObjectDistanceField,
)
from torch_robotics.torch_utils.torch_utils import to_numpy, DEFAULT_TENSOR_ARGS
from torch_robotics.trajectory.utils import interpolate_traj_via_points
from torch_robotics.visualizers.configuration_free_space import plot_configuration_free_space
from torch_robotics.visualizers.plot_utils import create_fig_and_axes, create_animation_video, plot_multiline


class Task(ABC):

    def __init__(self, env=None, robot=None, tensor_args=DEFAULT_TENSOR_ARGS, **kwargs):
        self.env = env
        self.robot = robot
        self.tensor_args = tensor_args


class PlanningTask(Task):

    def __init__(
        self,
        parametric_trajectory: ParametricTrajectoryBase,
        ws_limits=None,
        use_occupancy_map=False,
        cell_size=0.01,
        min_distance_robot_env=0.01,
        obstacle_cutoff_margin=0.01,
        margin_for_dense_collision_checking=0.0,
        use_field_collision_self=True,  # consider self collision
        use_field_collision_objects=True,  # consider object collision
        use_field_collision_ws_boundaries=False,  # consider workspace boundaries collision
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert parametric_trajectory is not None, "parametric_trajectory needs to be provided"
        self.parametric_trajectory = parametric_trajectory

        self.q_pos_start = torch.zeros((self.robot.q_dim,), **self.tensor_args)
        self.q_pos_goal = torch.zeros((self.robot.q_dim,), **self.tensor_args)
        self.set_q_pos_start_goal(self.q_pos_start, self.q_pos_goal)
        self.ee_pose_goal = torch.eye(4, **self.tensor_args)[:3, :]

        self.ws_limits = self.env.limits if ws_limits is None else ws_limits
        self.ws_min = self.ws_limits[0]
        self.ws_max = self.ws_limits[1]

        # Optional: use an occupancy map for collision checking -- useful for sampling-based algorithms
        # A precomputed collision map is faster when checking for collisions, in comparison to computing the distances
        # from tasks spaces to objects
        self.use_occupancy_map = use_occupancy_map
        if use_occupancy_map:
            self.env.build_occupancy_map(cell_size=cell_size)

        self.margin_for_dense_collision_checking = margin_for_dense_collision_checking

        self.min_distance_robot_env = min_distance_robot_env

        ################################################################################################
        # Collision fields
        # collision field for self-collision
        self.df_collision_self = self.robot.df_collision_self

        # collision field for objects
        self.df_collision_objects = CollisionObjectDistanceField(
            self.robot,
            df_obj_list_fn=self.env.get_df_obj_list,
            link_margins_for_object_collision_checking_tensor=self.robot.link_collision_spheres_radii,
            cutoff_margin=obstacle_cutoff_margin,
            tensor_args=self.tensor_args,
        )

        self.df_collision_extra_objects = None
        if self.env.obj_extra_list is not None:
            self.df_collision_extra_objects = CollisionObjectDistanceField(
                self.robot,
                df_obj_list_fn=partial(self.env.get_df_obj_list, return_extra_objects_only=True),
                link_margins_for_object_collision_checking_tensor=self.robot.link_collision_spheres_radii,
                cutoff_margin=obstacle_cutoff_margin,
                tensor_args=self.tensor_args,
            )
            self._collision_fields_extra_objects = [self.df_collision_extra_objects]
        else:
            self._collision_fields_extra_objects = []

        # collision field for workspace boundaries
        self.df_collision_ws_boundaries = CollisionWorkspaceBoundariesDistanceField(
            self.robot,
            link_margins_for_object_collision_checking_tensor=self.robot.link_collision_spheres_radii,
            cutoff_margin=obstacle_cutoff_margin,
            ws_bounds_min=self.ws_min,
            ws_bounds_max=self.ws_max,
            tensor_args=self.tensor_args,
        )

        self.df_collision_self = self.df_collision_self if use_field_collision_self else None

        self.df_collision_objects = self.df_collision_objects if use_field_collision_objects else None

        self.df_collision_ws_boundaries = self.df_collision_ws_boundaries if use_field_collision_ws_boundaries else None

        self._collision_fields = [
            self.df_collision_self,
            self.df_collision_objects,
            self.df_collision_ws_boundaries,
        ]

        ################################################################################################
        # Visualization
        self.colors = {"collision": "black", "free": "orange"}
        self.colors_robot = {"collision": "black", "free": "darkorange"}
        self.cmaps = {"collision": "Greys", "free": "Oranges"}
        self.cmaps_robot = {"collision": "Greys", "free": "YlOrRd"}

    def set_q_pos_start_goal(self, q_pos_start, q_pos_goal, **kwargs):
        self.q_pos_start = q_pos_start
        self.parametric_trajectory.q_pos_start = q_pos_start
        self.q_pos_goal = q_pos_goal
        self.parametric_trajectory.q_pos_goal = q_pos_goal

    def set_ee_pose_goal(self, ee_pose_goal, **kwargs):
        self.ee_pose_goal = ee_pose_goal

    def get_all_collision_fields(self):
        return [field for field in self._collision_fields if field is not None]

    def get_all_collision_fields_extra_objects(self):
        return [field for field in self._collision_fields_extra_objects if field is not None]

    def get_collision_self_field(self):
        return self.df_collision_self

    def get_collision_objects_field(self):
        return self.df_collision_objects

    def get_collision_extra_objects_field(self):
        return self.df_collision_extra_objects

    def get_collision_ws_boundaries_field(self):
        return self.df_collision_ws_boundaries

    def distance_q_pos(self, q1, q2):
        return self.robot.distance_q_pos(q1, q2)

    def sample_q_pos(self, without_collision=True, **kwargs):
        if without_collision:
            return self.random_coll_free_q_pos(**kwargs)
        else:
            return self.robot.random_q(**kwargs)

    def random_coll_free_q_pos(self, n_samples=1, max_samples=1000, max_tries=1000):
        # Random position in configuration space not in collision
        reject = True
        samples = torch.zeros((n_samples, self.robot.q_dim), **self.tensor_args)
        idx_begin = 0
        for i in range(max_tries):
            qs = self.robot.random_q(max_samples)
            in_collision = self.compute_collision(qs).squeeze()
            idxs_not_in_collision = torch.argwhere(in_collision == False).squeeze()
            if idxs_not_in_collision.nelement() == 0:
                # all points are in collision
                continue
            if idxs_not_in_collision.nelement() == 1:
                idxs_not_in_collision = [idxs_not_in_collision]
            idx_random = torch.randperm(len(idxs_not_in_collision))[:n_samples]
            free_qs = qs[idxs_not_in_collision[idx_random]]
            idx_end = min(idx_begin + free_qs.shape[0], samples.shape[0])
            samples[idx_begin:idx_end] = free_qs[: idx_end - idx_begin]
            idx_begin = idx_end
            if idx_end >= n_samples:
                reject = False
                break

        if reject:
            sys.exit("Could not find a collision free configuration")

        return samples.squeeze()

    def compute_collision(self, x, **kwargs):
        q_pos = self.get_position(x)
        return self._compute_collision_or_cost(q_pos, field_type="occupancy", **kwargs)

    def compute_collision_cost(self, x, **kwargs):
        q_pos = self.get_position(x)
        return self._compute_collision_or_cost(q_pos, field_type="sdf", **kwargs)

    def _compute_collision_or_cost(self, q_pos, field_type="occupancy", **kwargs):
        # q.shape needs to be reshaped to (batch, horizon, q_dim)
        q_original_shape = q_pos.shape
        b = 1
        h = 1
        collisions = None
        if q_pos.ndim == 1:
            q_pos = q_pos.unsqueeze(0).unsqueeze(0)  # add batch and horizon dimensions for interface
            collisions = torch.ones((1,), **self.tensor_args)
        elif q_pos.ndim == 2:
            b = q_pos.shape[0]
            q = q_pos.unsqueeze(1)  # add horizon dimension for interface
            collisions = torch.ones((b, 1), **self.tensor_args)  # (batch, 1)
        elif q_pos.ndim == 3:
            b = q_pos.shape[0]
            h = q_pos.shape[1]
            collisions = torch.ones((b, h), **self.tensor_args)  # (batch, horizon)
        elif q_pos.ndim > 3:
            raise NotImplementedError

        if self.use_occupancy_map:
            raise NotImplementedError
            # ---------------------------------- For occupancy maps ----------------------------------
            ########################################
            # Configuration space boundaries
            idxs_coll_free = torch.argwhere(
                torch.all(
                    torch.logical_and(
                        torch.greater_equal(q, self.robot.q_pos_min), torch.less_equal(q, self.robot.q_pos_max)
                    ),
                    dim=-1,
                )
            )  # I, 2

            # check if all points are out of bounds (in collision)
            if idxs_coll_free.nelement() == 0:
                return collisions

            ########################################
            # Task space collisions
            # forward kinematics
            q_try = q[idxs_coll_free[:, 0], idxs_coll_free[:, 1]]  # I, q_dim
            x_pos = self.robot.fk_map_collision(q_try, pos_only=True)  # I, taskspaces, x_dim

            # workspace boundaries
            # configuration is not valid if any points in the tasks spaces is out of workspace boundaries
            idxs_ws_in_boundaries = torch.argwhere(
                torch.all(
                    torch.all(
                        torch.logical_and(
                            torch.greater_equal(x_pos, self.ws_min), torch.less_equal(x_pos, self.ws_max)
                        ),
                        dim=-1,
                    ),
                    dim=-1,
                )
            ).squeeze()  # I_ws

            idxs_coll_free = idxs_coll_free[idxs_ws_in_boundaries].view(-1, 2)

            # collision in tasks space
            x_pos_in_ws = x_pos[idxs_ws_in_boundaries]  # I_ws, x_dim
            collisions_pos_x = self.env_tr.occupancy_map.get_collisions(x_pos_in_ws, **kwargs)
            if len(collisions_pos_x.shape) == 1:
                collisions_pos_x = collisions_pos_x.view(1, -1)
            idxs_taskspace = torch.argwhere(torch.all(collisions_pos_x == 0, dim=-1)).squeeze()

            idxs_coll_free = idxs_coll_free[idxs_taskspace].view(-1, 2)

            # filter collisions
            if len(collisions) == 1:
                collisions[idxs_coll_free[:, 0]] = 0
            else:
                collisions[idxs_coll_free[:, 0], idxs_coll_free[:, 1]] = 0
        else:
            # ---------------------------------- For distance fields ----------------------------------
            ########################################
            # For distance fields

            # forward kinematics
            fk_collision_pos = self.robot.fk_map_collision(q_pos)  # batch, horizon, taskspaces, x_dim

            ########################
            # Self collision
            if self.df_collision_self is not None:
                cost_collision_self = self.df_collision_self.compute_cost(
                    q_pos, fk_collision_pos, field_type=field_type, **kwargs
                )
            else:
                cost_collision_self = 0

            # Object collision
            if self.df_collision_objects is not None:
                cost_collision_objects = self.df_collision_objects.compute_cost(
                    q_pos, fk_collision_pos, field_type=field_type, **kwargs
                )
            else:
                cost_collision_objects = 0

            # Workspace boundaries
            if self.df_collision_ws_boundaries is not None:
                cost_collision_border = self.df_collision_ws_boundaries.compute_cost(
                    q_pos, fk_collision_pos, field_type=field_type, **kwargs
                )
            else:
                cost_collision_border = 0

            if field_type == "occupancy":
                collisions = cost_collision_self | cost_collision_objects | cost_collision_border
            else:
                collisions = cost_collision_self + cost_collision_objects + cost_collision_border

        return collisions

    def get_trajs_unvalid_and_valid(
        self, q_trajs, return_indices=False, num_interpolation=0, filter_joint_limits_vel_acc=False, **kwargs
    ):
        assert q_trajs.ndim == 3 or q_trajs.ndim == 4
        N = 1
        if q_trajs.ndim == 4:  # n_goals (or steps), batch of trajectories, length, dim
            N, B, H, D = q_trajs.shape
            trajs_new = einops.rearrange(q_trajs, "N B H D -> (N B) H D")
        else:
            B, H, D = q_trajs.shape
            trajs_new = q_trajs

        ###############################################################################################################
        # compute collisions on a finer-interpolated trajectory
        if num_interpolation > 0:
            trajs_interpolated = interpolate_traj_via_points(trajs_new, num_interpolation=num_interpolation)
        else:
            trajs_interpolated = trajs_new
        # Set a low margin for collision checking, which means we allow trajectories to pass very close to objects.
        # While the optimized trajectory via points are not at a 0 margin from the object, the interpolated via points
        # might be. A 0 margin guarantees that we do not discard those trajectories, while ensuring they are not in
        # collision (margin < 0).
        trajs_waypoints_collisions = self.compute_collision(
            trajs_interpolated, margin=self.margin_for_dense_collision_checking
        )

        if q_trajs.ndim == 4:
            trajs_waypoints_collisions = einops.rearrange(trajs_waypoints_collisions, "(N B) H -> N B H", N=N, B=B)

        trajs_valid_idxs = torch.argwhere(torch.logical_not(trajs_waypoints_collisions).all(dim=-1))
        trajs_unvalid_idxs = torch.argwhere(trajs_waypoints_collisions.any(dim=-1))

        ###############################################################################################################
        # Filter the trajectories that are not in collision and are inside the joint limits
        if trajs_valid_idxs.nelement() == 0:
            pass
        else:
            if q_trajs.ndim == 4:
                trajs_valid_tmp = q_trajs[trajs_valid_idxs[:, 0], trajs_valid_idxs[:, 1], ...]
            else:
                trajs_valid_tmp = q_trajs[trajs_valid_idxs.squeeze(), ...]

            trajs_valid_tmp_position = self.get_position(trajs_valid_tmp)
            check_list = [
                trajs_valid_tmp_position >= self.robot.q_pos_min,
                trajs_valid_tmp_position <= self.robot.q_pos_max,
            ]
            if filter_joint_limits_vel_acc:
                if self.robot.dq_max is not None:
                    trajs_valid_tmp_velocity = self.get_velocity(trajs_valid_tmp)
                    check_list.extend(
                        [trajs_valid_tmp_velocity >= -self.robot.dq_max, trajs_valid_tmp_velocity <= self.robot.dq_max]
                    )
                if self.robot.ddq_max is not None:
                    trajs_valid_tmp_acceleration = self.get_acceleration(trajs_valid_tmp)
                    check_list.extend(
                        [
                            trajs_valid_tmp_acceleration >= -self.robot.ddq_max,
                            trajs_valid_tmp_acceleration <= self.robot.ddq_max,
                        ]
                    )

            check_joint_limits = functools.reduce(torch.logical_and, check_list)
            trajs_valid_inside_joint_limits_idxs = check_joint_limits.all(dim=-1).all(dim=-1)
            trajs_valid_inside_joint_limits_idxs = torch.atleast_1d(trajs_valid_inside_joint_limits_idxs)
            trajs_valid_idxs_try = trajs_valid_idxs[torch.argwhere(trajs_valid_inside_joint_limits_idxs).squeeze()]
            if trajs_valid_idxs_try.nelement() == 0:
                trajs_unvalid_idxs = torch.cat((trajs_unvalid_idxs, trajs_valid_idxs), dim=0)
            else:
                trajs_valid_idxs_joint_limits = trajs_valid_idxs[
                    torch.argwhere(torch.logical_not(trajs_valid_inside_joint_limits_idxs)).squeeze()
                ]
                if trajs_valid_idxs_joint_limits.ndim == 1:
                    trajs_valid_idxs_joint_limits = trajs_valid_idxs_joint_limits[..., None]
                trajs_unvalid_idxs = torch.cat((trajs_unvalid_idxs, trajs_valid_idxs_joint_limits))
            trajs_valid_idxs = trajs_valid_idxs_try

        ###############################################################################################################
        # Return trajectories valid and unvalid
        if q_trajs.ndim == 4:
            trajs_valid = q_trajs[trajs_valid_idxs[:, 0], trajs_valid_idxs[:, 1], ...]
            if trajs_valid.ndim == 2:
                trajs_valid = trajs_valid.unsqueeze(0).unsqueeze(0)
            trajs_unvalid = q_trajs[trajs_unvalid_idxs[:, 0], trajs_unvalid_idxs[:, 1], ...]
            if trajs_unvalid.ndim == 2:
                trajs_unvalid = trajs_unvalid.unsqueeze(0).unsqueeze(0)
        else:
            trajs_valid = q_trajs[trajs_valid_idxs.squeeze(), ...]
            if trajs_valid.ndim == 2:
                trajs_valid = trajs_valid.unsqueeze(0)
            trajs_unvalid = q_trajs[trajs_unvalid_idxs.squeeze(), ...]
            if trajs_unvalid.ndim == 2:
                trajs_unvalid = trajs_unvalid.unsqueeze(0)

        if trajs_unvalid.nelement() == 0:
            trajs_unvalid = None
        if trajs_valid.nelement() == 0:
            trajs_valid = None

        if return_indices:
            return trajs_unvalid, trajs_unvalid_idxs, trajs_valid, trajs_valid_idxs, trajs_waypoints_collisions
        return trajs_unvalid, trajs_valid

    def compute_fraction_valid_trajs(self, q_trajs, **kwargs):
        # Compute the fractions of trajs that are collision free
        _, trajs_unvalid_idxs, _, trajs_valid_idxs, _ = self.get_trajs_unvalid_and_valid(
            q_trajs, return_indices=True, **kwargs
        )
        n_trajs_valid = trajs_valid_idxs.nelement()
        n_trajs_unvalid = trajs_unvalid_idxs.nelement()
        return n_trajs_valid / (n_trajs_valid + n_trajs_unvalid)

    def compute_collision_intensity_trajs(self, q_trajs, **kwargs):
        # Compute the fraction of waypoints that are in collision
        _, _, _, _, trajs_waypoints_collisions = self.get_trajs_unvalid_and_valid(
            q_trajs, return_indices=True, **kwargs
        )
        return torch.count_nonzero(trajs_waypoints_collisions) / trajs_waypoints_collisions.nelement()

    def compute_success_valid_trajs(self, q_trajs, **kwargs):
        # If at least one trajectory is valid, then we consider success
        _, trajs_valid = self.get_trajs_unvalid_and_valid(q_trajs, **kwargs)
        if trajs_valid is not None:
            if trajs_valid.nelement() >= 1:
                return 1
        return 0

    ###############################################################################################################
    ###############################################################################################################
    # Parse state
    def get_position(self, x):
        return self.robot.get_position(x)

    def get_velocity(self, x, **kwargs):
        vel = self.robot.get_velocity(x)
        # If there is no velocity in the state
        if x.nelement() != 0 and vel.nelement() == 0:
            vel = None
        return vel

    def get_acceleration(self, x, dt=None):
        acc = self.robot.get_acceleration(x)
        # If there is no acceleration in the state
        if x.nelement() != 0 and acc.nelement() == 0:
            acc = None
        return acc

    ###############################################################################################################
    ###############################################################################################################
    # Visualisation
    def render_robot_trajectories(
        self, fig=None, ax=None, q_pos_trajs=None, q_pos_trajs_best=None, color_collisions=True, **kwargs
    ):
        if fig is None or ax is None:
            fig, ax = create_fig_and_axes(dim=self.env.dim)

        self.env.render(ax)
        if q_pos_trajs is not None:
            if color_collisions:
                _, q_trajs_coll_idxs, _, q_trajs_free_idxs, _ = self.get_trajs_unvalid_and_valid(
                    q_pos_trajs, return_indices=True, **kwargs
                )
                kwargs["colors"] = []
                for i in range(len(q_trajs_coll_idxs) + len(q_trajs_free_idxs)):
                    kwargs["colors"].append(self.colors["collision"] if i in q_trajs_coll_idxs else self.colors["free"])
            else:
                kwargs["colors"] = [self.colors["free"]] * len(q_pos_trajs)
        self.robot.render_trajectories(ax, q_pos_trajs=q_pos_trajs, **kwargs)
        if q_pos_trajs_best is not None:
            kwargs["colors"] = ["blue"]
            self.robot.render_trajectories(ax, q_pos_trajs=q_pos_trajs_best.unsqueeze(0), **kwargs)

        return fig, ax

    def animate_robot_trajectories(
        self,
        q_pos_trajs=None,
        q_pos_start=None,
        q_pos_goal=None,
        plot_x_trajs=False,
        n_frames=10,
        remove_title=False,
        process_axes=lambda x: x,
        **kwargs,
    ):
        if q_pos_trajs is None:
            return

        assert q_pos_trajs.ndim == 3
        B, H, D = q_pos_trajs.shape

        idxs = np.round(np.linspace(0, H - 1, n_frames)).astype(int)
        q_pos_trajs_selection = q_pos_trajs[:, idxs, :]

        fig, ax = create_fig_and_axes(dim=self.env.dim)

        def animate_fn(i, ax):
            ax.clear()
            if not remove_title:
                ax.set_title(f"step: {idxs[i]}/{H-1}")
            if plot_x_trajs:
                self.render_robot_trajectories(
                    fig=fig, ax=ax, q_pos_trajs=q_pos_trajs, q_pos_start=q_pos_start, q_pos_goal=q_pos_goal, **kwargs
                )
            else:
                self.env.render(ax)

            # TODO - implement batched version
            qs = q_pos_trajs_selection[:, i, :]  # batch, q_dim
            if qs.ndim == 1:
                qs = qs.unsqueeze(0)  # interface (batch, q_dim)
            for q in qs:
                self.robot.render(
                    ax,
                    q_pos=q,
                    color=(
                        self.colors_robot["collision"]
                        if self.compute_collision(q, margin=0.0)
                        else self.colors_robot["free"]
                    ),
                    arrow_length=0.1,
                    arrow_alpha=0.5,
                    arrow_linewidth=1.0,
                    cmap=self.cmaps["collision"] if self.compute_collision(q, margin=0.0) else self.cmaps["free"],
                    **kwargs,
                )

            if q_pos_start is not None:
                self.robot.render(ax, q_pos_start, color="blue", cmap="Greens", **kwargs)
            if q_pos_goal is not None:
                self.robot.render(ax, q_pos_goal, color="red", cmap="Purples", **kwargs)

            process_axes(ax)

        create_animation_video(fig, animate_fn, n_frames=n_frames, fargs=(ax,), **kwargs)

    def animate_opt_iters_robots(
        self,
        trajs_pos=None,
        traj_pos_best=None,
        start_state=None,
        goal_state=None,
        control_points=None,
        n_frames=10,
        remove_axes_labels_and_ticks=False,
        **kwargs,
    ):
        # trajs: steps, batch, horizon, q_dim
        if trajs_pos is None:
            return

        assert trajs_pos.ndim == 4
        S, B, H, D = trajs_pos.shape

        idxs = np.round(np.linspace(0, S - 1, n_frames)).astype(int)
        trajs_pos_selection = trajs_pos[idxs]
        if control_points is None:
            # Assume the control points are the trajectory waypoints
            control_points_selection = trajs_pos_selection
        else:
            control_points_selection = control_points[idxs]

        fig, ax = create_fig_and_axes(dim=self.env.dim)

        def animate_fn(i, ax):
            ax.clear()
            ax.set_title(f"iter: {idxs[i]}/{S-1}")
            self.render_robot_trajectories(
                fig=fig,
                ax=ax,
                q_pos_trajs=trajs_pos_selection[i],
                control_points=control_points_selection[i],
                q_pos_trajs_best=traj_pos_best if i == n_frames - 1 else None,
                start_state=start_state,
                goal_state=goal_state,
                **kwargs,
            )
            if start_state is not None:
                self.robot.render(ax, start_state, color="green", cmap="Greens")
            if goal_state is not None:
                self.robot.render(ax, goal_state, color="purple", cmap="Purples")

            if remove_axes_labels_and_ticks:
                remove_axes_labels_ticks(ax)

        create_animation_video(fig, animate_fn, n_frames=n_frames, fargs=(ax,), **kwargs)

    def animate_opt_iters_joint_space_env(
        self,
        trajs_pos=None,
        traj_pos_best=None,
        start_state=None,
        goal_state=None,
        control_points=None,
        n_frames=10,
        plot_control_points=False,
        **kwargs,
    ):
        # trajs: steps, batch, horizon, q_dim
        if trajs_pos is None:
            return

        assert trajs_pos.ndim == 4
        S, B, H, D = trajs_pos.shape

        idxs = np.round(np.linspace(0, S - 1, n_frames)).astype(int)
        trajs_selection = trajs_pos[idxs]
        if control_points is None:
            # Assume the control points are the trajectory waypoints
            control_points_selection = trajs_selection
        else:
            control_points_selection = control_points[idxs]

        fig, ax = create_fig_and_axes(dim=2)

        def animate_fn(i, fig=fig, ax=ax):
            ax.clear()

            fig, ax = plot_configuration_free_space(
                self,
                fig=fig,
                ax=ax,
                q_dim0=0,
                q_dim1=1,
                N_meshgrid=1000,
                smooth_sigma=5.0,
                use_imshow=False,
            )

            ax.set_title(f"iter: {idxs[i]}/{S-1}")
            ax.set_xlim(self.robot.q_pos_min_np[0], self.robot.q_pos_max_np[0])
            ax.set_ylim(self.robot.q_pos_min_np[1], self.robot.q_pos_max_np[1])

            start_state_np = to_numpy(start_state)
            ax.scatter(start_state_np[0], start_state_np[1], color="blue", marker="X", s=10**2.9, zorder=100)
            goal_state_np = to_numpy(goal_state)
            ax.scatter(goal_state_np[0], goal_state_np[1], color="red", marker="X", s=10**2.9, zorder=100)

            if plot_control_points:
                control_points = control_points_selection[i]
                control_points_np = to_numpy(control_points)
                ax.scatter(
                    control_points_np[..., 0].reshape(-1),
                    control_points_np[..., 1].reshape(-1),
                    color="blue",
                    marker="o",
                    s=10**2.0,
                    zorder=100,
                )

            trajs_unvalid, trajs_valid = self.get_trajs_unvalid_and_valid(trajs_selection[i])
            if trajs_unvalid is not None:
                for traj in trajs_unvalid:
                    traj_np = to_numpy(traj)
                    ax.plot(
                        traj_np[:, 0], traj_np[:, 1], color="black", linewidth=2.0, alpha=1.0, zorder=10, linestyle="-"
                    )
            if trajs_valid is not None:
                for traj in trajs_valid:
                    traj_np = to_numpy(traj)
                    ax.plot(
                        traj_np[:, 0], traj_np[:, 1], color="orange", linewidth=2.0, alpha=1.0, zorder=10, linestyle="-"
                    )

            # best trajectory
            if traj_pos_best is not None and i == n_frames - 1:
                traj_best_np = to_numpy(traj_pos_best)
                ax.plot(
                    traj_best_np[:, 0],
                    traj_best_np[:, 1],
                    color="blue",
                    linewidth=2.0,
                    alpha=1.0,
                    zorder=15,
                    linestyle="-",
                )

        create_animation_video(fig, animate_fn, n_frames=n_frames, **kwargs)

    def plot_joint_space_trajectories(
        self,
        fig=None,
        axs=None,
        q_pos_trajs=None,
        q_vel_trajs=None,
        q_acc_trajs=None,
        q_pos_traj_best=None,
        q_vel_traj_best=None,
        q_acc_traj_best=None,
        q_pos_start=None,
        q_pos_goal=None,
        q_vel_start=None,
        q_vel_goal=None,
        q_acc_start=None,
        q_acc_goal=None,
        set_q_pos_limits=True,
        set_q_vel_limits=True,
        set_q_acc_limits=True,
        control_points=None,
        **kwargs,
    ):
        if q_pos_trajs is None:
            return
        q_pos_trajs_np = to_numpy(q_pos_trajs)

        assert q_pos_trajs_np.ndim == 3  # batch, horizon, q dimension

        # Filter trajectories not in collision and inside joint limits
        q_trajs_l = [q_pos_trajs]
        if q_vel_trajs is not None:
            q_trajs_l.append(q_vel_trajs)
        if q_acc_trajs is not None:
            q_trajs_l.append(q_acc_trajs)
        q_trajs = torch.cat(q_trajs_l, dim=-1)

        q_trajs_coll, q_trajs_coll_idxs, q_trajs_free, q_trajs_free_idxs, _ = self.get_trajs_unvalid_and_valid(
            q_trajs, return_indices=True, **kwargs
        )

        q_pos_trajs_coll_np = None
        q_vel_trajs_coll_np = None
        q_acc_trajs_coll_np = None
        if q_trajs_coll is not None:
            q_pos_trajs_coll_np = to_numpy(self.get_position(q_trajs_coll))
            if q_pos_trajs_coll_np.ndim == 2:
                q_pos_trajs_coll_np = q_pos_trajs_coll_np[None, ...]
            if q_vel_trajs is not None:
                q_vel_trajs_coll_np = to_numpy(self.get_velocity(q_trajs_coll))
                if q_vel_trajs_coll_np.ndim == 2:
                    q_vel_trajs_coll_np = q_vel_trajs_coll_np[None, ...]
            if q_acc_trajs is not None:
                q_acc_trajs_coll_np = to_numpy(self.get_acceleration(q_trajs_coll))
                if q_acc_trajs_coll_np.ndim == 2:
                    q_acc_trajs_coll_np = q_acc_trajs_coll_np[None, ...]

        q_pos_trajs_free_np = None
        q_vel_trajs_free_np = None
        q_acc_trajs_free_np = None
        if q_trajs_free is not None:
            q_pos_trajs_free_np = to_numpy(self.get_position(q_trajs_free))
            if q_pos_trajs_free_np.ndim == 2:
                q_pos_trajs_free_np = q_pos_trajs_free_np[None, ...]
            if q_vel_trajs is not None:
                q_vel_trajs_free_np = to_numpy(self.get_velocity(q_trajs_free))
                if q_vel_trajs_free_np.ndim == 2:
                    q_vel_trajs_free_np = q_vel_trajs_free_np[None, ...]
            if q_acc_trajs is not None:
                q_acc_trajs_free_np = to_numpy(self.get_acceleration(q_trajs_free))
                if q_acc_trajs_free_np.ndim == 2:
                    q_acc_trajs_free_np = q_acc_trajs_free_np[None, ...]

        if q_pos_start is not None:
            q_pos_start = to_numpy(q_pos_start)
        if q_vel_start is not None:
            q_vel_start = to_numpy(q_vel_start)
        if q_acc_start is not None:
            q_acc_start = to_numpy(q_acc_start)
        if q_pos_goal is not None:
            q_pos_goal = to_numpy(q_pos_goal)
        if q_vel_goal is not None:
            q_vel_goal = to_numpy(q_vel_goal)
        if q_acc_goal is not None:
            q_acc_goal = to_numpy(q_acc_goal)

        if fig is None or axs is None:
            fig, axs = plt.subplots(self.robot.q_dim, 3, squeeze=False, figsize=(18, 2.5 * self.robot.q_dim))

        axs[0, 0].set_title("Position")
        axs[0, 1].set_title("Velocity")
        axs[0, 2].set_title("Acceleration")
        axs[-1, 1].set_xlabel("Time [s]")
        timesteps = to_numpy(self.parametric_trajectory.get_timesteps().reshape(1, -1))
        t_start, t_goal = timesteps[0, 0], timesteps[0, -1]
        for i, ax in enumerate(axs):
            for q_trajs_filtered, color in zip(
                [
                    (q_pos_trajs_coll_np, q_vel_trajs_coll_np, q_acc_trajs_coll_np),
                    (q_pos_trajs_free_np, q_vel_trajs_free_np, q_acc_trajs_free_np),
                ],
                ["black", "orange"],
            ):
                # Positions, velocities, accelerations
                for j, q_trajs_filtered_item in enumerate(q_trajs_filtered):
                    if q_trajs_filtered_item is not None:
                        plot_multiline(
                            ax[j],
                            np.repeat(timesteps, q_trajs_filtered_item.shape[0], axis=0),
                            q_trajs_filtered_item[..., i],
                            color=color,
                            **kwargs,
                        )

            if q_pos_traj_best is not None:
                q_pos_traj_best_np = to_numpy(q_pos_traj_best)
                plot_multiline(ax[0], timesteps, q_pos_traj_best_np[..., i].reshape(1, -1), color="blue", **kwargs)
            if q_vel_traj_best is not None:
                q_vel_traj_best_np = to_numpy(q_vel_traj_best)
                plot_multiline(ax[1], timesteps, q_vel_traj_best_np[..., i].reshape(1, -1), color="blue", **kwargs)
            if q_acc_traj_best is not None:
                q_acc_traj_best_np = to_numpy(q_acc_traj_best)
                plot_multiline(ax[2], timesteps, q_acc_traj_best_np[..., i].reshape(1, -1), color="blue", **kwargs)

            # Start and goal
            for j, x in enumerate([q_pos_start, q_vel_start, q_acc_start]):
                if x is not None:
                    ax[j].scatter(t_start, x[i], color="green")

            for j, x in enumerate([q_pos_goal, q_vel_goal, q_acc_goal]):
                if x is not None:
                    ax[j].scatter(t_goal, x[i], color="purple")

            ax[0].set_ylabel(f"$q_{i}$")

            if set_q_pos_limits:
                q_pos_min, q_pos_max = self.robot.q_pos_min_np[i], self.robot.q_pos_max_np[i]
                padding = 0.1 * np.abs(q_pos_max - q_pos_min)
                ax[0].set_ylim(q_pos_min - padding, q_pos_max + padding)
                ax[0].plot([t_start, t_goal], [q_pos_max, q_pos_max], color="k", linestyle="--")
                ax[0].plot([t_start, t_goal], [q_pos_min, q_pos_min], color="k", linestyle="--")
            if set_q_vel_limits and self.robot.dq_max_np is not None:
                ax[1].plot(
                    [t_start, t_goal], [self.robot.dq_max_np[i], self.robot.dq_max_np[i]], color="k", linestyle="--"
                )
                ax[1].plot(
                    [t_start, t_goal], [-self.robot.dq_max_np[i], -self.robot.dq_max_np[i]], color="k", linestyle="--"
                )
            if set_q_acc_limits and self.robot.ddq_max_np is not None:
                ax[2].plot(
                    [t_start, t_goal], [self.robot.ddq_max_np[i], self.robot.ddq_max_np[i]], color="k", linestyle="--"
                )
                ax[2].plot(
                    [t_start, t_goal], [-self.robot.ddq_max_np[i], -self.robot.ddq_max_np[i]], color="k", linestyle="--"
                )

        # time limits
        t_eps = 0.1
        for ax in list(itertools.chain(*axs)):
            ax.set_xlim(t_start - t_eps, t_goal + t_eps)

        # plot control points
        if control_points is not None:
            control_points_np = to_numpy(control_points)
            control_points_timesteps = to_numpy(self.parametric_trajectory.get_phase_steps())
            for control_points_np_one in control_points_np:
                for i, ax in enumerate(axs):
                    ax[0].scatter(control_points_timesteps, control_points_np_one[:, i], color="red", s=2**2, zorder=10)

        return fig, axs

    def animate_opt_iters_joint_space_state(
        self,
        q_pos_trajs=None,
        q_vel_trajs=None,
        q_acc_trajs=None,
        q_pos_traj_best=None,
        q_vel_traj_best=None,
        q_acc_traj_best=None,
        n_frames=10,
        **kwargs,
    ):
        # trajs: steps, batch, horizon, q_dim
        if q_pos_trajs is None:
            return

        assert q_pos_trajs.ndim == 4
        S, B, H, D = q_pos_trajs.shape

        idxs = np.round(np.linspace(0, S - 1, n_frames)).astype(int)
        q_pos_trajs_selection = q_pos_trajs[idxs]
        q_vel_trajs_selection = None
        if q_vel_trajs is not None:
            q_vel_trajs_selection = q_vel_trajs[idxs]
        q_acc_trajs_selection = None
        if q_acc_trajs is not None:
            q_acc_trajs_selection = q_acc_trajs[idxs]

        fig, axs = self.plot_joint_space_trajectories(
            q_pos_trajs=q_pos_trajs_selection[0],
            q_vel_trajs=q_vel_trajs_selection[0] if q_vel_trajs is not None else None,
            q_acc_trajs=q_acc_trajs_selection[0] if q_acc_trajs is not None else None,
            **kwargs,
        )

        def animate_fn(i):
            if i == n_frames - 1:
                print()
            [ax.clear() for ax in axs.ravel()]
            fig.suptitle(f"iter: {idxs[i]}/{S-1}")
            self.plot_joint_space_trajectories(
                fig=fig,
                axs=axs,
                q_pos_trajs=q_pos_trajs_selection[i],
                q_vel_trajs=q_vel_trajs_selection[i] if q_vel_trajs is not None else None,
                q_acc_trajs=q_acc_trajs_selection[i] if q_acc_trajs is not None else None,
                q_pos_traj_best=q_pos_traj_best if i == n_frames - 1 else None,
                q_vel_traj_best=q_vel_traj_best if i == n_frames - 1 else None,
                q_acc_traj_best=q_acc_traj_best if i == n_frames - 1 else None,
                **kwargs,
            )

        create_animation_video(fig, animate_fn, n_frames=n_frames, **kwargs)
