from abc import ABC, abstractmethod

import einops
import numpy as np
import torch

from mp_baselines.planners.chomp import CHOMP
from mp_baselines.planners.costs.factors.field_factor import FieldFactor
from mp_baselines.planners.costs.factors.gp_factor import GPFactor
from mp_baselines.planners.costs.factors.unary_factor import UnaryFactor
from torch_robotics.torch_utils.torch_utils import batched_weighted_dot_prod, DEFAULT_TENSOR_ARGS
from torch_robotics.trajectory.utils import finite_difference_vector, interpolate_points_v1


class Cost(ABC):
    def __init__(self, robot, n_support_points, tensor_args=DEFAULT_TENSOR_ARGS, **kwargs):
        self.robot = robot
        self.n_dof = robot.q_dim
        self.dim = 2 * self.n_dof  # position + velocity
        self.n_support_points = n_support_points

        self.tensor_args = tensor_args

    def set_cost_factors(self):
        pass

    def __call__(self, trajs, **kwargs):
        return self.eval(trajs, **kwargs)

    @abstractmethod
    def eval(self, trajs, **kwargs):
        pass

    @abstractmethod
    def get_linear_system(self, trajs, **kwargs):
        pass

    def get_q_pos_vel_and_fk_map(self, trajs, **kwargs):
        assert trajs.ndim == 3 or trajs.ndim == 4
        N = 1
        if trajs.ndim == 4:
            N, B, H, D = trajs.shape  # n_goals (or steps), batch of trajectories, length, dim
            trajs = einops.rearrange(trajs, "N B H D -> (N B) H D")
        else:
            B, H, D = trajs.shape

        q_pos = self.robot.get_position(trajs)
        q_vel = self.robot.get_velocity(trajs)
        H_positions = self.robot.fk_map_collision(
            q_pos
        )  # I, taskspaces, x_dim+1, x_dim+1 (homogeneous transformation matrices)
        return trajs, q_pos, q_vel, H_positions


class CostComposite(Cost):

    def __init__(self, robot, n_support_points, cost_list, weights_cost_l=None, **kwargs):
        super().__init__(robot, n_support_points, **kwargs)
        self.cost_l = cost_list
        self.weight_cost_l = weights_cost_l if weights_cost_l is not None else [1.0] * len(cost_list)

    def eval(self, trajs, trajs_interpolated=None, return_invidual_costs_and_weights=False, **kwargs):
        trajs, q_pos, q_vel, H_positions = self.get_q_pos_vel_and_fk_map(trajs)

        if not return_invidual_costs_and_weights:
            cost_total = 0
            for cost, weight_cost in zip(self.cost_l, self.weight_cost_l):
                if trajs_interpolated is not None:
                    # Compute only collision costs with interpolated trajectories.
                    # Other costs are computed with non-interpolated trajectories, e.g. smoothness
                    if isinstance(cost, CostCollision):
                        trajs_tmp = trajs_interpolated
                    else:
                        trajs_tmp = trajs
                else:
                    trajs_tmp = trajs
                cost_tmp = weight_cost * cost(trajs_tmp, q_pos=q_pos, q_vel=q_vel, H_positions=H_positions, **kwargs)
                cost_total += cost_tmp
            return cost_total
        else:
            cost_l = []
            for cost in self.cost_l:
                if trajs_interpolated is not None:
                    # Compute only collision costs with interpolated trajectories.
                    # Other costs are computed with non-interpolated trajectories, e.g. smoothness
                    if isinstance(cost, CostCollision):
                        trajs_tmp = trajs_interpolated
                    else:
                        trajs_tmp = trajs
                else:
                    trajs_tmp = trajs

                cost_tmp = cost(trajs_tmp, q_pos=q_pos, q_vel=q_vel, H_positions=H_positions, **kwargs)
                cost_l.append(cost_tmp)

            if return_invidual_costs_and_weights:
                return cost_l, self.weight_cost_l

    def get_linear_system(self, trajs, n_interpolated_points=None, **kwargs):
        trajs.requires_grad = True
        # TODO - join fk map into one call for trajs and trajs_interp
        trajs, q_pos, q_vel, H_positions = self.get_q_pos_vel_and_fk_map(trajs)

        # Upsample trajectory for finer collision checking
        # Interpolate in joint space
        # TODO - change from linear interpolation to GP interpolation
        if n_interpolated_points is None:
            trajs_interp, q_pos_interp, q_vel_interp, H_positions_interp = None, None, None, None
        else:
            trajs_interp = interpolate_points_v1(trajs, n_interpolated_points)
            trajs_interp, q_pos_interp, q_vel_interp, H_positions_interp = self.get_q_pos_vel_and_fk_map(trajs_interp)

        batch_size = trajs.shape[0]
        As, bs, Ks = [], [], []
        optim_dim = 0
        for cost, weight_cost in zip(self.cost_l, self.weight_cost_l):
            A, b, K = cost.get_linear_system(
                trajs,
                q_pos=q_pos,
                q_vel=q_vel,
                H_positions=H_positions,
                trajs_interp=trajs_interp,
                q_pos_interp=q_pos_interp,
                q_vel_interp=q_vel_interp,
                H_positions_interp=H_positions_interp,
                **kwargs,
            )
            if A is None or b is None or K is None:
                continue
            optim_dim += A.shape[1]
            As.append(A.detach())
            bs.append(b.detach())
            Ks.append(K.detach())

        A = torch.cat(As, dim=1)
        b = torch.cat(bs, dim=1)
        K = torch.zeros(batch_size, optim_dim, optim_dim, **self.tensor_args)
        offset = 0
        for i in range(len(Ks)):
            dim = Ks[i].shape[1]
            K[:, offset : offset + dim, offset : offset + dim] = Ks[i]
            offset += dim
        return A, b, K


class CostCollision(Cost):

    def __init__(self, robot, n_support_points, field=None, sigma_coll=None, **kwargs):
        super().__init__(robot, n_support_points, **kwargs)
        self.field = field
        self.sigma_coll = sigma_coll

        self.set_cost_factors()

    def set_cost_factors(self):
        # ========= Cost factors ===============
        self.obst_factor = FieldFactor(
            self.n_dof, self.sigma_coll, [1, None]  # take the whole trajectory except for the first point
        )

    def eval(self, trajs, q_pos=None, q_vel=None, H_positions=None, **observation):
        costs = 0
        if self.field is not None:
            # H_pos = link_pos_from_link_tensor(H)  # get translation part from transformation matrices
            H_pos = H_positions
            err_obst = self.obst_factor.get_error(
                trajs,
                self.field,
                q_pos=q_pos,
                q_vel=q_vel,
                H_pos=H_pos,
                calc_jacobian=False,  # TODO: NOTE(an): no need for grads in StochGPMP
                obstacle_spheres=observation.get("obstacle_spheres", None),
            )
            w_mat = self.obst_factor.K
            obst_costs = w_mat * err_obst.sum(1)
            costs = obst_costs

        return costs

    def get_linear_system(
        self,
        trajs,
        q_pos=None,
        q_vel=None,
        H_positions=None,
        trajs_interp=None,
        q_pos_interp=None,
        q_vel_interp=None,
        H_positions_interp=None,
        **observation,
    ):
        A, b, K = None, None, None
        if self.field is not None:
            batch_size = trajs.shape[0]
            # H_pos = link_pos_from_link_tensor(H)  # get translation part from transformation matrices
            # H_pos = link_pos_from_link_tensor(H)  # get translation part from transformation matrices
            H_pos = H_positions

            # Get H_obst wrt to the interpolated trajectory. This is computed inside get_error

            err_obst, H_obst = self.obst_factor.get_error(
                trajs,
                self.field,
                q_pos=q_pos,
                q_vel=q_vel,
                H_pos=H_pos,
                trajs_interp=trajs_interp,
                q_pos_interp=q_pos_interp,
                q_vel_interp=q_vel_interp,
                H_pos_interp=H_positions_interp,
                calc_jacobian=True,
                obstacle_spheres=observation.get("obstacle_spheres", None),
            )

            A = torch.zeros(batch_size, self.n_support_points - 1, self.dim * self.n_support_points, **self.tensor_args)
            A[:, :, : H_obst.shape[-1]] = H_obst
            # shift each row by self.dim
            idxs = torch.arange(A.shape[-1], **self.tensor_args).repeat(A.shape[-2], 1)
            idxs = (
                idxs - torch.arange(self.dim, (idxs.shape[0] + 1) * self.dim, self.dim, **self.tensor_args).view(-1, 1)
            ) % idxs.shape[-1]
            idxs = idxs.to(torch.int64)
            A = torch.gather(A, -1, idxs.repeat(batch_size, 1, 1))

            # old code not vectorized
            # https://github.com/anindex/stoch_gpmp/blob/main/stoch_gpmp/costs/cost_functions.py#L275

            b = err_obst.unsqueeze(-1)
            K = self.obst_factor.K * torch.eye((self.n_support_points - 1), **self.tensor_args).repeat(batch_size, 1, 1)

        return A, b, K


class CostGP(Cost):

    def __init__(self, robot, n_support_points, start_state, dt, sigma_params, **kwargs):
        super().__init__(robot, n_support_points, **kwargs)
        self.start_state = start_state
        self.dt = dt

        self.sigma_start = sigma_params["sigma_start"]
        self.sigma_gp = sigma_params["sigma_gp"]

        self.set_cost_factors()

    def set_cost_factors(self):
        # ========= Cost factors ===============
        self.start_prior = UnaryFactor(
            self.dim,
            self.sigma_start,
            self.start_state,
            self.tensor_args,
        )

        self.gp_prior = GPFactor(
            self.n_dof,
            self.sigma_gp,
            self.dt,
            self.n_support_points - 1,
            self.tensor_args,
        )

    def eval(self, trajs, **observation):
        # trajs = trajs.reshape(-1, self.n_support_points, self.dim)
        # Start cost
        err_p = self.start_prior.get_error(trajs[:, [0]], calc_jacobian=False)
        w_mat = self.start_prior.K
        start_costs = err_p @ w_mat.unsqueeze(0) @ err_p.transpose(1, 2)
        start_costs = start_costs.squeeze()

        # GP Trajectory cost
        err_gp = self.gp_prior.get_error(trajs, calc_jacobian=False)
        w_mat = self.gp_prior.Q_inv[0]  # repeated Q_inv
        w_mat = w_mat.reshape(1, 1, self.dim, self.dim)
        gp_costs = err_gp.transpose(2, 3) @ w_mat @ err_gp
        gp_costs = gp_costs.sum(1)
        gp_costs = gp_costs.squeeze()

        costs = start_costs + gp_costs

        return costs

    def get_linear_system(self, trajs, **observation):
        batch_size = trajs.shape[0]
        A = torch.zeros(
            batch_size, self.dim * self.n_support_points, self.dim * self.n_support_points, **self.tensor_args
        )
        b = torch.zeros(batch_size, self.dim * self.n_support_points, 1, **self.tensor_args)
        K = torch.zeros(
            batch_size, self.dim * self.n_support_points, self.dim * self.n_support_points, **self.tensor_args
        )

        # Start prior factor
        err_p, H_p = self.start_prior.get_error(trajs[:, [0]])
        A[:, : self.dim, : self.dim] = H_p
        b[:, : self.dim] = err_p
        K[:, : self.dim, : self.dim] = self.start_prior.K

        # GP factors
        err_gp, H1_gp, H2_gp = self.gp_prior.get_error(trajs)

        A[:, self.dim :, : -self.dim] = torch.block_diag(*H1_gp)
        A[:, self.dim :, self.dim :] += torch.block_diag(*H2_gp)
        b[:, self.dim :] = einops.rearrange(err_gp, "b h d 1 -> b (h d) 1")
        K[:, self.dim :, self.dim :] += torch.block_diag(*self.gp_prior.Q_inv)

        # old code not vectorized
        # https://github.com/anindex/stoch_gpmp/blob/main/stoch_gpmp/costs/cost_functions.py#L161

        return A, b, K


class CostGPTrajectory(Cost):

    def __init__(self, robot, n_support_points, dt, sigma_gp=None, **kwargs):
        super().__init__(robot, n_support_points, **kwargs)
        self.dt = dt

        self.sigma_gp = sigma_gp

        self.set_cost_factors()

    def set_cost_factors(self):
        # ========= Cost factors ===============
        self.gp_prior = GPFactor(
            self.n_dof,
            self.sigma_gp,
            self.dt,
            self.n_support_points - 1,
            self.tensor_args,
        )

    def eval(self, trajs, **observation):
        # trajs = trajs.reshape(-1, self.n_support_points, self.dim)

        # GP cost
        err_gp = self.gp_prior.get_error(trajs, calc_jacobian=False)
        w_mat = self.gp_prior.Q_inv[0]  # repeated Q_inv
        w_mat = w_mat.reshape(1, 1, self.dim, self.dim)
        gp_costs = err_gp.transpose(2, 3) @ w_mat @ err_gp
        gp_costs = gp_costs.sum(1).squeeze()
        costs = gp_costs
        return costs

    def get_linear_system(self, trajs, **observation):
        pass


class CostGPTrajectoryPositionOnlyWrapper(CostGPTrajectory):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def eval(self, trajs, **observation):
        vel = finite_difference_vector(trajs, dt=self.dt, method="central")
        trajs_tmp = torch.cat((trajs, vel), dim=-1)
        return super().eval(trajs_tmp, **observation)


class CostSmoothnessCHOMP(Cost):

    def __init__(self, robot, n_support_points, **kwargs):
        super().__init__(robot, n_support_points, **kwargs)
        self.dt = robot.dt

        self.Sigma_inv = CHOMP._get_R_mat(dt=self.dt, n_support_points=n_support_points, **kwargs)

    def eval(self, trajs, **observation):
        R_mat = self.Sigma_inv
        cost = batched_weighted_dot_prod(trajs, R_mat, trajs)
        return cost

    def get_linear_system(self, trajs, **observation):
        pass


class CostJointLimits(Cost):

    def __init__(self, robot, n_support_points, eps=np.deg2rad(3), **kwargs):
        super().__init__(robot, n_support_points, **kwargs)

        self.eps = eps

    def eval(self, trajs, **observation):
        assert trajs.ndim == 3

        # trajs = trajs.reshape(-1, self.n_support_points, self.dim)
        trajs_pos = self.robot.get_position(trajs)

        idxs_lower = torch.argwhere(trajs_pos < self.robot.q_pos_min + self.eps)
        cost_lower = torch.pow(
            self.robot.q_pos_min[idxs_lower[:, 2]]
            + self.eps
            - trajs_pos[idxs_lower[:, 0], idxs_lower[:, 1], idxs_lower[:, 2]],
            2,
        ).sum(-1)

        idxs_upper = torch.argwhere(trajs_pos > self.robot.q_pos_max - self.eps)
        cost_upper = torch.pow(
            self.robot.q_pos_max[idxs_upper[:, 2]]
            - self.eps
            - trajs_pos[idxs_upper[:, 0], idxs_upper[:, 1], idxs_upper[:, 2]],
            2,
        ).sum(-1)

        costs = cost_lower + cost_upper

        return costs

    def get_linear_system(self, trajs, **observation):
        pass


class CostGoal(Cost):

    def __init__(self, robot, n_support_points, field=None, sigma_goal=None, **kwargs):
        super().__init__(robot, n_support_points, **kwargs)
        self.field = field
        self.sigma_goal = sigma_goal

        self.set_cost_factors()

    def set_cost_factors(self):
        # ========= Cost factors ===============
        self.goal_factor = FieldFactor(self.n_dof, self.sigma_goal, [-1, None])  # only take last point

    def eval(self, trajs, x_trajs=None, **observation):
        costs = 0
        if self.field is not None:
            err_obst = self.goal_factor.get_error(
                trajs,
                self.field,
                x_trajs=x_trajs,
                calc_jacobian=False,  # NOTE(an): no need for grads in StochGPMP
            )
            w_mat = self.goal_factor.K
            obst_costs = w_mat * err_obst.sum(1)
            costs = obst_costs

        return costs

    def get_linear_system(self, trajs, x_trajs=None, **observation):
        A, b, K = None, None, None
        if self.field is not None:
            batch_size = trajs.shape[0]
            A = torch.zeros(batch_size, 1, self.dim * self.n_support_points, **self.tensor_args)
            err_goal, H_goal = self.goal_factor.get_error(
                trajs,
                self.field,
                x_trajs=x_trajs,
                calc_jacobian=True,
            )
            A[:, :, -self.dim : (-self.dim + self.n_dof)] = H_goal
            b = err_goal.unsqueeze(-1)
            K = self.goal_factor.K * torch.eye(1, **self.tensor_args).repeat(batch_size, 1, 1)
        return A, b, K


class CostGoalPrior(Cost):

    def __init__(
        self,
        robot,
        n_support_points,
        multi_goal_states=None,  # num_goal x n_dim (pos + vel)
        num_particles_per_goal=None,
        num_samples=None,
        sigma_goal_prior=None,
        **kwargs,
    ):
        super().__init__(robot, n_support_points, **kwargs)
        self.multi_goal_states = multi_goal_states
        self.num_goals = multi_goal_states.shape[0]
        self.num_particles_per_goal = num_particles_per_goal
        self.num_particles = num_particles_per_goal * self.num_goals
        self.num_samples = num_samples
        self.sigma_goal_prior = sigma_goal_prior

        self.set_cost_factors()

    def set_cost_factors(self):
        self.multi_goal_prior = []
        # TODO: remove this for loop
        for i in range(self.num_goals):
            self.multi_goal_prior.append(
                UnaryFactor(
                    self.dim,
                    self.sigma_goal_prior,
                    self.multi_goal_states[i],
                    self.tensor_args,
                )
            )

    def eval(self, trajs, **observation):
        costs = 0
        if self.multi_goal_states is not None:
            x = trajs.reshape(
                self.num_goals, self.num_particles_per_goal * self.num_samples, self.n_support_points, self.dim
            )
            costs = torch.zeros(self.num_goals, self.num_particles_per_goal * self.num_samples, **self.tensor_args)
            # TODO: remove this for loop
            for i in range(self.num_goals):
                err_g = self.multi_goal_prior[i].get_error(x[i, :, [-1]], calc_jacobian=False)
                w_mat = self.multi_goal_prior[i].K
                goal_costs = err_g @ w_mat.unsqueeze(0) @ err_g.transpose(1, 2)
                goal_costs = goal_costs.squeeze()
                costs[i] += goal_costs
            costs = costs.flatten()
        return costs

    def get_linear_system(self, trajs, **observation):
        A, b, K = None, None, None
        if self.multi_goal_states is not None:
            npg = self.num_particles_per_goal
            batch_size = npg * self.num_goals
            x = trajs.reshape(self.num_goals, self.num_particles_per_goal, self.n_support_points, self.dim)
            A = torch.zeros(batch_size, self.dim, self.dim * self.n_support_points, **self.tensor_args)
            b = torch.zeros(batch_size, self.dim, 1, **self.tensor_args)
            K = torch.zeros(batch_size, self.dim, self.dim, **self.tensor_args)
            # TODO: remove this for loop
            for i in range(self.num_goals):
                err_g, H_g = self.multi_goal_prior[i].get_error(x[i, :, [-1]])
                A[i * npg : (i + 1) * npg, :, -self.dim :] = H_g
                b[i * npg : (i + 1) * npg] = err_g
                K[i * npg : (i + 1) * npg] = self.multi_goal_prior[i].K

        return A, b, K
