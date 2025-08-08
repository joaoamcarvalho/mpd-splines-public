import time
from functools import partial

import einops
import torch
from dotmap import DotMap

from deps.theseus.torchlie.torchlie.functional.se3_impl import _adjoint_impl
from mpd.parametric_trajectory.trajectory_waypoints import ParametricTrajectoryWaypoints
from torch_robotics.torch_kinematics_tree.geometrics.utils import link_pos_from_link_tensor
from torch_robotics.torch_utils.torch_timer import TimerCUDA

from torchlie.functional import SE3 as SE3_Func

from torch.func import vmap, jacrev, functional_call

from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, to_numpy
from torch_robotics.visualizers.plot_utils import create_fig_and_axes


def project_hierarchical_gradients_fast(grads):
    """
    Faster implementation of hierarchical gradient projection that properly
    preserves higher priority constraints.

    Args:
        grads: List of gradients, highest priority first
    Returns:
        final_grad: Sum of projected gradients
        grads_projected_l: List of individual projected gradients
    """
    if len(grads) == 1:
        return grads[0], grads

    # Pre-compute normalized high-priority gradients
    grads_stack_flatten = einops.rearrange(torch.stack(grads), "... h d -> ... (h d)")
    norms = torch.norm(grads_stack_flatten, dim=-1, keepdim=True)
    normalized_grads = grads_stack_flatten / (norms + 1e-8)

    # Initialize with the highest priority gradient
    grads_projected_l = [
        einops.rearrange(grads_stack_flatten[0], "... (h d) -> ... h d", h=grads[0].shape[-2], d=grads[0].shape[-1])
    ]

    # For each lower priority gradient
    for i in range(1, len(grads_stack_flatten)):
        curr_grad = grads_stack_flatten[i]

        # Project sequentially through all higher priority gradients
        for j in range(i):
            # Create projector for this constraint
            n = normalized_grads[j]
            projection = torch.einsum("...i,...j->...ij", n, n)
            # Apply projection
            curr_grad = curr_grad - torch.einsum("...ij,...j->...i", projection, curr_grad)

        grads_projected_l.append(
            einops.rearrange(curr_grad, "... (h d) -> ... h d", h=grads[0].shape[-2], d=grads[0].shape[-1])
        )

    # Sum all projected gradients
    final_grad = torch.stack(grads_projected_l).sum(dim=0)

    return final_grad, grads_projected_l


class NoCostException(Exception):
    pass


class CostGuideManagerParametricTrajectory:

    def __init__(self, planning_task, dataset, args_inference, tensor_args=DEFAULT_TENSOR_ARGS, debug=False, **kwargs):
        self.args_inference = args_inference

        self.tensor_args = tensor_args

        self.planning_task = planning_task
        self.env = planning_task.env
        self.robot = planning_task.robot
        self.parametric_trajectory = planning_task.parametric_trajectory

        self.dataset = dataset

        # control points normalization function gradients
        self.q_cps_unnormalize_fn = lambda x: self.dataset.unnormalize_control_points(x).sum()
        self.grad_cps_wrt_cps_normalized = self.dataset.grad_unnormalized_wrt_control_points_normalized

        # Setup costs
        self.costs = DotMap()
        self.setup_costs()
        if not self.costs:
            raise NoCostException

        self.step_guide_call = 0
        self._t = 0

        self.debug = debug

    def setup_costs(self):
        for cost_key in self.args_inference.costs:
            if cost_key == "CostTaskSpaceEEGoalPose" and not self.dataset.context_ee_goal_pose:
                # Skip the cost if the EE goal context is not set. This means the last joint position is fixed, so the
                # EE pose is determined via FK.
                continue
            try:
                cost_options = self.args_inference.costs[cost_key]
                cost = eval(cost_key)(self.planning_task, **cost_options)
                self.costs[cost_key] = DotMap()
                self.costs[cost_key].cost = cost
                self.costs[cost_key].weight = cost_options.weight
            except NoCostException:
                continue

    def use_all_collision_objects(self):
        """
        Set the cost to use all collision objects.
        """
        cost_key = "CostTaskSpaceCollisionObjects"
        if cost_key in self.costs:
            self.costs[cost_key].cost.use_only_on_extra_objects = False

    @torch.enable_grad()
    def __call__(self, control_points_normalized, return_cost=False, warmup=False, plot_gradients=False, **kwargs):
        """
        Args:
            control_points_normalized: (batch_size, n_control_points, q_dim)
        """
        if self.debug:
            print()
            print(f"Guide step {self.step_guide_call}")

        # Unnormalize the control points.
        # The generative model outputs normalized control points, but the costs are defined on the unnormalized
        # trajectory space.
        control_points = self.dataset.unnormalize_control_points(control_points_normalized)

        # Get the trajectory (position, velocity, acceleration) from the control points, in phase.
        control_points.requires_grad_(True)
        q_traj_in_phase_d = self.parametric_trajectory.get_q_trajectory(
            control_points, None, None, get_type=("pos", "vel", "acc"), get_time_representation=False
        )
        q_traj_pos_in_phase = q_traj_in_phase_d["pos"]
        q_traj_vel_in_phase = q_traj_in_phase_d["vel"]
        q_traj_acc_in_phase = q_traj_in_phase_d["acc"]

        # Compute forward kinematics and spatial (world) jacobians
        assert q_traj_pos_in_phase.ndim == 3
        q_traj_pos_in_phase_original_shape = q_traj_pos_in_phase.shape
        q_traj_pos_aux = einops.rearrange(q_traj_pos_in_phase, "... d -> (...) d")

        with TimerCUDA() as t_fk_jac:
            # collision links and jacobians
            jacs_spatial, link_poses = self.robot.jfk_s_collision_spheres(q_traj_pos_aux)
            jacs_spatial_th = torch.stack(jacs_spatial).transpose(
                0, 1
            )  # ((batch_size, traejectory_length), n_links, 6, d)
            jacs_spatial_th = einops.rearrange(
                jacs_spatial_th, "(b h) ... -> b h ...", b=q_traj_pos_in_phase_original_shape[0]
            )
            link_poses_th = torch.stack(link_poses).transpose(0, 1)  # ((batch_size, traejectory_length), n_links, 3, 4)
            link_poses_th = einops.rearrange(
                link_poses_th, "(b h) ... -> b h ...", b=q_traj_pos_in_phase_original_shape[0]
            )

            # end effector links and jacobians
            jacs_spatial_ee, link_poses_ee = self.robot.jfk_s_ee(q_traj_pos_aux)
            jacs_spatial_th_ee = torch.stack(jacs_spatial_ee).transpose(
                0, 1
            )  # ((batch_size, traejectory_length), n_links, 6, d)
            jacs_spatial_th_ee = einops.rearrange(
                jacs_spatial_th_ee, "(b h) ... -> b h ...", b=q_traj_pos_in_phase_original_shape[0]
            )
            link_poses_th_ee = torch.stack(link_poses_ee).transpose(
                0, 1
            )  # ((batch_size, traejectory_length), n_links, 3, 4)
            link_poses_th_ee = einops.rearrange(
                link_poses_th_ee, "(b h) ... -> b h ...", b=q_traj_pos_in_phase_original_shape[0]
            )

        if self.debug:
            print(f"FK and Jacobians (time): {t_fk_jac.elapsed:.4f} s")
            print("-" * 50)

        # Compute cost and gradients wrt to the control points normalized
        with TimerCUDA() as t_cost_grad_all:
            cost_all = 0.0
            grad_costs_wrt_cp_normalized_l = []
            rs_inv = self.parametric_trajectory.phase_time.rs_inv
            s = self.parametric_trajectory.phase_time.s
            for k, cost_key in enumerate(self.costs):
                s_time = time.perf_counter()

                cost_fn = self.costs[cost_key].cost
                weight = self.costs[cost_key].weight

                cost_single_in_phase, grad_cost_single_wrt_cp_normalized_in_phase = (
                    self.compute_cost_grad_cp_normalized(
                        cost_fn,
                        control_points_normalized,
                        control_points,
                        q_traj_pos_in_phase,
                        q_traj_vel_in_phase,
                        q_traj_acc_in_phase,
                        link_poses_th,
                        jacs_spatial_th,
                        link_poses_th_ee,
                        jacs_spatial_th_ee,
                    )
                )

                if self.dataset.context_ee_goal_pose and cost_key != "CostTaskSpaceEEGoalPose":
                    # If the EE pose goal context is set, the generative model determines the last joint position.
                    # Hence, we zero the gradient of the last control point for all costs except the EE pose goal cost,
                    # to avoid changing the last joint position.
                    # Note that the penultimate control point of the B-spline still affects the last portion of
                    # the trajectory, which means it can remove it from collisions, even though the last control point
                    # is fixed.
                    grad_cost_single_wrt_cp_normalized_in_phase[..., -1, :] = 0.0

                if self.dataset.context_ee_goal_pose and cost_key == "CostTaskSpaceEEGoalPose":
                    # The CostTaskSpaceEEGoalPose cost is defined only on the last point of the trajectory,
                    # so we use directly the cost and gradient at the last point, without integration.
                    cost_single = cost_single_in_phase[..., -1]
                    grad_cost_single_wrt_cp_normalized = grad_cost_single_wrt_cp_normalized_in_phase[:, -1, ...]
                else:
                    # Approximate integral in eq. 28 -- https://arxiv.org/pdf/2412.19948
                    cost_single = torch.trapezoid(
                        cost_single_in_phase * rs_inv,
                        s,
                        dim=-1,
                    )

                    grad_cost_single_wrt_cp_normalized = torch.trapezoid(
                        grad_cost_single_wrt_cp_normalized_in_phase * rs_inv[None, :, None, None],
                        s,
                        dim=-3,
                    )

                cost_single_weighted = weight * cost_single
                cost_all += cost_single_weighted
                grad_cost_single_wrt_cp_normalized_weighted = weight * grad_cost_single_wrt_cp_normalized
                grad_costs_wrt_cp_normalized_l.append(grad_cost_single_wrt_cp_normalized_weighted)

                if self.debug:
                    print(f"{cost_key} (cost): {cost_single_weighted.mean():.4f} +- {cost_single_weighted.std():.4f}")
                    grad_cost_all_wrt_cp_normalized_norm_weighted = torch.linalg.norm(
                        grad_cost_single_wrt_cp_normalized_weighted, dim=-1
                    )
                    print(
                        f"{cost_key} (grad norm):"
                        f" {grad_cost_all_wrt_cp_normalized_norm_weighted.mean():.4f}"
                        f" +- {grad_cost_all_wrt_cp_normalized_norm_weighted.std():.4f}"
                    )
                    print(f"{cost_key} (time): {time.perf_counter() - s_time:.4f} s")
                    print(f"--------------------------------")

        if self.debug:
            print(f"Costs and gradients (time): {t_cost_grad_all.elapsed:.4f} s")

        # Project gradients respecting hierarchy
        if self.args_inference.project_gradient_hierarchy:
            with TimerCUDA() as t_project_gradients:
                grad_costs_all_wrt_cp_normalized, grad_costs_all_wrt_cp_normalized_projected_l = (
                    project_hierarchical_gradients_fast(grad_costs_wrt_cp_normalized_l)
                )
            if self.debug:
                print(f"Project gradients (time): {t_project_gradients.elapsed:.4f} s")
        else:
            grad_costs_all_wrt_cp_normalized = torch.stack(grad_costs_wrt_cp_normalized_l).sum(dim=0)

        # -1 because the denoising gradient methods expect an objective function to maximize, but we want to minimize
        # the cost
        grad_costs_all_wrt_cp_normalized = -1.0 * grad_costs_all_wrt_cp_normalized

        # scatter plot of gradients in 2D
        if plot_gradients and self.debug and control_points_normalized.shape[-1] == 2:
            import matplotlib.pyplot as plt

            fig, ax = create_fig_and_axes(self.planning_task.env.dim)
            ax.scatter(to_numpy(control_points)[..., 0], to_numpy(control_points[..., 1]))
            self.planning_task.env.render(ax)
            self.planning_task.robot.render_trajectories(
                ax, q_traj_pos_in_phase, plot_points_scatter=False, control_points=control_points
            )

            grad_costs_wrt_cp_normalized_l += [-1.0 * grad_costs_all_wrt_cp_normalized]
            colors = ["g", "y", "c", "m", "k"]
            for k, grad in enumerate(grad_costs_wrt_cp_normalized_l):
                grad *= -1  # flip the gradient direction
                ax.quiver(
                    to_numpy(control_points[..., 0]),
                    to_numpy(control_points[..., 1]),
                    to_numpy(grad[..., 0]),
                    to_numpy(grad[..., 1]),
                    color=colors[k % len(colors)] if k < len(grad_costs_wrt_cp_normalized_l) - 1 else "blue",
                    label=f"{list(self.costs.keys())[k]}" if k < len(grad_costs_wrt_cp_normalized_l) - 1 else "Total",
                    scale=25,
                    width=0.005,
                )
            ax.legend()
            plt.show()

        # Increment step counter
        if not warmup:
            self.step_guide_call += 1
        if return_cost:
            return cost_all, grad_costs_all_wrt_cp_normalized
        return grad_costs_all_wrt_cp_normalized

    def compute_cost_grad_cp_normalized(
        self,
        cost_fn,
        control_points_normalized,
        control_points,
        q_traj_pos_in_phase,
        q_traj_vel_in_phase,
        q_traj_acc_in_phase,
        link_poses_th,
        jacs_spatial_th,
        link_poses_th_ee,
        jacs_spatial_th_ee,
        **kwargs,
    ):
        # compute cost gradients wrt to the control points normalized
        # The cost C can be a function of the trajectory q(s), the control points cp, or the task space x.
        # dC/dcp_norm = dC/dx * dx/dq * dq/dcp * dcp/dcp_normalized

        # We compute the gradient of the cost wrt to the joint space q
        # For TaskSpace costs, we compute dC/dq = dC/dx * dx/dq * dq/dcp
        # For JointSpace costs, we compute dC/dq = dC/dq * dq/dcp
        cost_value_in_phase, grad_cost_wrt_cp_in_phase = cost_fn.compute_cost_grad_wrt_cp(
            control_points,
            q_traj_pos_in_phase,
            q_traj_vel_in_phase,
            q_traj_acc_in_phase,
            link_poses_th,
            jacs_spatial_th,
            link_poses_th_ee,
            jacs_spatial_th_ee,
            **kwargs,
        )

        # Gradient of the control points wrt to the control points normalized
        # dcp/dcp_norm
        grad_cp_wrt_cp_normalized = self.grad_cps_wrt_cps_normalized(control_points_normalized)

        # Gradient of the cost wrt to the control points normalized
        # dC/dcp_norm = dC/dcp * dcp/dcp_norm
        # In matrix form -- Hadamard product (the normalization is done element-wise)
        grad_cost_wrt_cp_normalized_per_shape_step = torch.einsum(
            "...jkn,...kn->...jkn", grad_cost_wrt_cp_in_phase, grad_cp_wrt_cp_normalized
        )

        return cost_value_in_phase, grad_cost_wrt_cp_normalized_per_shape_step

    def warmup(self, shape_x, **kwargs):
        x = torch.randn(shape_x, **self.tensor_args)
        self.__call__(x, warmup=True)


class CostSpace:

    def __init__(self, planning_task, tensor_args=DEFAULT_TENSOR_ARGS, **kwargs):
        self.planning_task = planning_task
        self.parametric_trajectory = planning_task.parametric_trajectory
        self.robot = planning_task.robot
        self.env = planning_task.env
        self.tensor_args = tensor_args

    def compute_cost_grad_wrt_cp(self, *args, **kwargs):
        raise NotImplementedError

    def compute_grad_cost_wrt_cp(
        self,
        control_points,
        grad_cost_wrt_q,
        get_type_single="pos",
    ):
        # Gradient of the joint space trajectory position wrt to the control points
        # dq/dcp
        grad_q_pos_wrt_cp = self.parametric_trajectory.get_grad_q_traj_in_phase_wrt_control_points(
            control_points,
            get_type=(get_type_single,),
            remove_control_points=True,
        )[get_type_single]
        # Gradient of cost wrt to the control points per phase step
        # dC/dcp = dC/dq * dq/dcp
        # In matrix form dC/dcp = (dq/dcp)^T @ dC/dq
        try:
            grad_cost_wrt_cp = torch.einsum("...ihk,...hn->...hkn", grad_q_pos_wrt_cp, grad_cost_wrt_q)
        except RuntimeError:
            # for waypoints, we sum over the state dimension
            grad_cost_wrt_cp = torch.einsum("...hdkn,...hn->...hkn", grad_q_pos_wrt_cp, grad_cost_wrt_q)

        return grad_cost_wrt_cp


class CostJointSpace(CostSpace):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CostJointSpaceJointLimits(CostJointSpace):

    def __init__(self, planning_task, eps=0.03, **kwargs):
        super().__init__(planning_task, **kwargs)
        self.eps = eps
        self.q_min = self.robot.q_pos_min
        self.q_max = self.robot.q_pos_max
        if self.robot.dq_max is not None:
            self.dq_min = -self.robot.dq_max
            self.dq_max = self.robot.dq_max
        else:
            self.dq_min = None
            self.dq_max = None
        if self.robot.ddq_max is not None:
            self.ddq_min = -self.robot.ddq_max
            self.ddq_max = self.robot.ddq_max
        else:
            self.ddq_min = None
            self.ddq_max = None

    def compute_cost_grad_wrt_cp(
        self, control_points, q_traj_pos_in_phase, q_traj_vel_in_phase, q_traj_acc_in_phase, *args, **kwargs
    ):
        # positions
        # transform the joint position limits from time to phase
        q_min_in_phase = self.q_min + self.eps
        q_max_in_phase = self.q_max - self.eps
        mask_low = torch.less_equal(q_traj_pos_in_phase, q_min_in_phase)
        mask_high = torch.greater_equal(q_traj_pos_in_phase, q_max_in_phase)

        cost_pos_limit_low = 0.5 * torch.linalg.norm((q_min_in_phase - q_traj_pos_in_phase) * mask_low, dim=-1)
        cost_pos_limit_high = 0.5 * torch.linalg.norm((q_max_in_phase - q_traj_pos_in_phase) * mask_high, dim=-1)
        cost_pos_limit = cost_pos_limit_low + cost_pos_limit_high
        grad_cost_q_pos_low = mask_low * (q_min_in_phase - q_traj_pos_in_phase) * -1
        grad_cost_q_pos_high = mask_high * (q_max_in_phase - q_traj_pos_in_phase) * -1
        grad_cost_wrt_q_pos = grad_cost_q_pos_low + grad_cost_q_pos_high

        grad_cost_pos_wrt_cp = self.compute_grad_cost_wrt_cp(
            control_points,
            grad_cost_wrt_q_pos,
            get_type_single="pos",
        )

        # velocities
        rs_inv = self.parametric_trajectory.phase_time.rs_inv
        rs = self.parametric_trajectory.phase_time.rs
        dr_ds = self.parametric_trajectory.phase_time.dr_ds

        cost_vel_limit = 0.0
        grad_cost_vel_wrt_cp = 0.0
        if self.dq_max is not None:
            # transform the joint velocity limits from time to phase
            dq_min_in_phase = (self.dq_min + self.eps) * rs_inv[..., None]
            dq_max_in_phase = (self.dq_max - self.eps) * rs_inv[..., None]
            mask_low = torch.less_equal(q_traj_vel_in_phase, dq_min_in_phase)
            mask_high = torch.greater_equal(q_traj_vel_in_phase, dq_max_in_phase)

            cost_vel_limit_low = 0.5 * torch.linalg.norm((q_min_in_phase - q_traj_vel_in_phase) * mask_low, dim=-1)
            cost_vel_limit_high = 0.5 * torch.linalg.norm((q_max_in_phase - q_traj_vel_in_phase) * mask_high, dim=-1)
            cost_vel_limit = cost_vel_limit_low + cost_vel_limit_high
            grad_cost_q_vel_low = mask_low * (q_min_in_phase - q_traj_vel_in_phase) * -1
            grad_cost_q_vel_high = mask_high * (q_max_in_phase - q_traj_vel_in_phase) * -1
            grad_cost_wrt_q_vel = grad_cost_q_vel_low + grad_cost_q_vel_high

            grad_cost_vel_wrt_cp = self.compute_grad_cost_wrt_cp(
                control_points,
                grad_cost_wrt_q_vel,
                get_type_single="vel",
            )

        # accelerations
        cost_acc_limit = 0.0
        grad_cost_acc_wrt_cp = 0.0
        if self.ddq_max is not None:
            # transform the joint acceleration limits from time to phase
            ddq_min_in_phase = (
                self.ddq_min + self.eps - q_traj_vel_in_phase * dr_ds[..., None] * rs[..., None]
            ) * rs_inv[..., None] ** 2
            ddq_max_in_phase = (
                self.ddq_max - self.eps - q_traj_vel_in_phase * dr_ds[..., None] * rs[..., None]
            ) * rs_inv[..., None] ** 2
            mask_low = torch.less_equal(q_traj_acc_in_phase, ddq_min_in_phase)
            mask_high = torch.greater_equal(q_traj_acc_in_phase, ddq_max_in_phase)

            cost_acc_limit_low = 0.5 * torch.linalg.norm((ddq_min_in_phase - q_traj_acc_in_phase) * mask_low, dim=-1)
            cost_acc_limit_high = 0.5 * torch.linalg.norm((ddq_max_in_phase - q_traj_acc_in_phase) * mask_high, dim=-1)
            cost_acc_limit = cost_acc_limit_low + cost_acc_limit_high
            grad_cost_q_acc_low = mask_low * (ddq_min_in_phase - q_traj_acc_in_phase) * -1
            grad_cost_q_acc_high = mask_high * (ddq_max_in_phase - q_traj_acc_in_phase) * -1
            grad_cost_wrt_q_acc = grad_cost_q_acc_low + grad_cost_q_acc_high

            grad_cost_acc_wrt_cp = self.compute_grad_cost_wrt_cp(
                control_points,
                grad_cost_wrt_q_acc,
                get_type_single="acc",
            )

        return (
            cost_pos_limit + cost_vel_limit + cost_acc_limit,
            grad_cost_pos_wrt_cp + grad_cost_vel_wrt_cp + grad_cost_acc_wrt_cp,
        )


class CostJointSpacePathLength(CostJointSpace):

    def __init__(self, planning_task, **kwargs):
        super().__init__(planning_task, **kwargs)

    def compute_cost_grad_wrt_cp(
        self, control_points, q_traj_pos_in_phase, q_traj_vel_in_phase, q_traj_acc_in_phase, *args, **kwargs
    ):
        q_traj_pos_in_phase.requires_grad_(True)
        q_traj_pos_diff = torch.zeros_like(q_traj_pos_in_phase)
        q_traj_pos_diff[..., 1:, :] = torch.diff(q_traj_pos_in_phase, dim=-2)
        cost_pos = 0.5 * torch.linalg.norm(q_traj_pos_diff, dim=-1)
        grad_cost_wrt_q_pos = torch.autograd.grad(cost_pos.sum(), [q_traj_pos_in_phase], retain_graph=True)[0]

        grad_cost_pos_wrt_cp = self.compute_grad_cost_wrt_cp(
            control_points,
            grad_cost_wrt_q_pos,
            get_type_single="pos",
        )

        return cost_pos, grad_cost_pos_wrt_cp


class CostJointSpaceVelocity(CostJointSpace):

    def __init__(self, planning_task, **kwargs):
        super().__init__(planning_task, **kwargs)

    def compute_cost_grad_wrt_cp(
        self, control_points, q_traj_pos_in_phase, q_traj_vel_in_phase, q_traj_acc_in_phase, *args, **kwargs
    ):
        q_traj_vel_in_phase.requires_grad_(True)
        cost_vel = 0.5 * torch.linalg.norm(q_traj_vel_in_phase, dim=-1)
        grad_cost_wrt_q_vel = torch.autograd.grad(cost_vel.sum(), [q_traj_vel_in_phase], retain_graph=True)[0]

        grad_cost_vel_wrt_cp = self.compute_grad_cost_wrt_cp(
            control_points,
            grad_cost_wrt_q_vel,
            get_type_single="vel",
        )

        return cost_vel, grad_cost_vel_wrt_cp


class CostJointSpaceAcceleration(CostJointSpace):

    def __init__(self, planning_task, **kwargs):
        super().__init__(planning_task, **kwargs)

    def compute_cost_grad_wrt_cp(
        self, control_points, q_traj_pos_in_phase, q_traj_vel_in_phase, q_traj_acc_in_phase, *args, **kwargs
    ):
        q_traj_acc_in_phase.requires_grad_(True)
        cost_acc = 0.5 * torch.linalg.norm(q_traj_acc_in_phase, dim=-1)
        grad_cost_wrt_q_acc = torch.autograd.grad(cost_acc.sum(), [q_traj_acc_in_phase], retain_graph=True)[0]

        grad_cost_acc_wrt_cp = self.compute_grad_cost_wrt_cp(
            control_points,
            grad_cost_wrt_q_acc,
            get_type_single="acc",
        )

        return cost_acc, grad_cost_acc_wrt_cp


class CostTaskSpace(CostSpace):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_cost_grad_wrt_cp(self, *args, **kwargs):
        raise NotImplementedError

    def get_jacobians_position(self, jac_spatial):
        return jac_spatial[..., : self.robot.task_space_dim, :]

    def get_jacobians_orientation(self, jac_spatial):
        # the starting index is 3, because the first 3 elements are the position
        return jac_spatial[..., 3:, :]


def map_jacobian_from_world_to_local_world_aligned(link_pose_w, jacobian_spatial):
    # map jacobian from world (spatial) to local world aligned using the Adjoint matrix
    H_lwa_w = link_pose_w.clone()
    H_lwa_w[..., :3, :3] = torch.eye(3, dtype=link_pose_w.dtype, device=link_pose_w.device)
    H_lwa_w[..., :3, 3] = -1 * H_lwa_w[..., :3, 3]
    Ad_lwa_w = _adjoint_impl(H_lwa_w)
    jac_lwa = Ad_lwa_w @ jacobian_spatial
    return jac_lwa


class CostTaskSpaceCollisionObjects(CostTaskSpace):
    def __init__(self, planning_task, use_only_on_extra_objects=False, **kwargs):
        super().__init__(planning_task, **kwargs)
        self._use_only_on_extra_objects = use_only_on_extra_objects
        self.collision_objects_field = None
        self.update_collision_objects_field()

    @property
    def use_only_on_extra_objects(self):
        return self._use_only_on_extra_objects

    @use_only_on_extra_objects.setter
    def use_only_on_extra_objects(self, val):
        """
        If True, the cost will be computed only on the extra collision objects.
        If False, the cost will be computed on all collision objects.
        """
        self._use_only_on_extra_objects = val
        self.update_collision_objects_field()

    def update_collision_objects_field(self):
        if self._use_only_on_extra_objects:
            self.collision_objects_field = self.planning_task.get_collision_extra_objects_field()
        else:
            self.collision_objects_field = self.planning_task.get_collision_objects_field()
        if self.collision_objects_field is None:
            raise NoCostException

    def compute_cost_grad_wrt_cp(
        self,
        control_points,
        q_traj_pos_in_phase,
        q_traj_vel_in_phase,
        q_traj_acc_in_phase,
        x_poses,
        jacobians_spatial,
        *args,
        **kwargs,
    ):
        if self.collision_objects_field is None:
            return 0.0, 0.0

        # get link positions
        x_positions = link_pos_from_link_tensor(x_poses)[..., : self.robot.task_space_dim]

        # C, dC/dx
        # cost (and gradient) of q trajectory in phase space
        # derivative of eq. 28 wrt control points -- https://arxiv.org/pdf/2412.19948
        cost, grad_cost_wrt_x = self.collision_objects_field.compute_distance_field_cost_and_gradient(x_positions)

        # (dx/dq)^T @ dC/dx (jacobian transpose x task space error)
        # map jacobian from world (spatial) to local world aligned using the adjoint
        # the SDF is defined in the world frame, hence we need to map the jacobian to the local world aligned frame
        jacs_lwa = map_jacobian_from_world_to_local_world_aligned(x_poses, jacobians_spatial)
        jacs_lwa_position = self.get_jacobians_position(jacs_lwa)
        grad_cost_wrt_q_pos = torch.einsum("...dj,...d->...j", jacs_lwa_position, grad_cost_wrt_x)

        # sum the cost and gradient over the task space links
        cost = cost.sum(dim=-1)
        grad_cost_wrt_q_pos = grad_cost_wrt_q_pos.sum(dim=-2)

        grad_cost_wrt_cp = self.compute_grad_cost_wrt_cp(
            control_points,
            grad_cost_wrt_q_pos,
            get_type_single="pos",
        )

        return cost, grad_cost_wrt_cp


class CostTaskSpaceCollisionSelf(CostTaskSpace):
    def __init__(self, planning_task, **kwargs):
        super().__init__(planning_task, **kwargs)
        self.collision_self_field = self.planning_task.get_collision_self_field()
        if self.collision_self_field is None:
            raise NoCostException

    def compute_cost_grad_wrt_cp(
        self,
        control_points,
        q_traj_pos_in_phase,
        q_traj_vel_in_phase,
        q_traj_acc_in_phase,
        x_poses,
        jacobians_spatial,
        *args,
        **kwargs,
    ):
        # get link positions
        x_positions = link_pos_from_link_tensor(x_poses)[..., : self.robot.task_space_dim]

        # C, dC/dx
        with torch.enable_grad():
            x_positions.requires_grad_(True)
            cost, _ = self.collision_self_field.compute_distance_field_cost_and_gradient(x_positions)
            # TODO - implement gradient inside the collision self field
            grad_cost_wrt_x = torch.autograd.grad(cost.sum(), [x_positions])[0]

        # (dx/dq)^T @ dC/dx (jacobian transpose x task space error)
        # map jacobian from world (spatial) to local world aligned using the adjoint
        # the self collision distance is defined in the world frame, hence we need to map the jacobian to the
        # local world aligned frame
        jacs_lwa = map_jacobian_from_world_to_local_world_aligned(x_poses, jacobians_spatial)
        jacs_lwa_position = self.get_jacobians_position(jacs_lwa)
        grad_cost_wrt_q_pos = torch.einsum("...dj,...d->...j", jacs_lwa_position, grad_cost_wrt_x)

        # sum the cost and gradient over the task space links
        cost = cost  # the cost is the max absolute self collision distance
        grad_cost_wrt_q_pos = grad_cost_wrt_q_pos.sum(dim=-2)

        grad_cost_wrt_cp = self.compute_grad_cost_wrt_cp(
            control_points,
            grad_cost_wrt_q_pos,
            get_type_single="pos",
        )

        return cost, grad_cost_wrt_cp


class CostTaskSpaceEEGoalPose(CostTaskSpace):
    def __init__(self, planning_task, **kwargs):
        super().__init__(planning_task, **kwargs)

    def compute_cost_grad_wrt_cp(
        self,
        control_points,
        q_traj_pos_in_phase,
        q_traj_vel_in_phase,
        q_traj_acc_in_phase,
        link_poses_th,
        jacs_spatial_th,
        link_poses_th_ee,
        jacs_spatial_th_ee,
        *args,
        **kwargs,
    ):
        ee_pose_goal = self.planning_task.ee_pose_goal
        assert ee_pose_goal is not None, "The end effector goal pose is not set in planning_task."

        # last n points of the trajectory
        ee_pose = link_poses_th_ee.squeeze(-3)

        # pose error in tangent space se(3)
        # error = log(W_EE_goal * W_EE_current^-1)
        ee_pose_inv = SE3_Func.inv(ee_pose)
        error = SE3_Func.log(SE3_Func.compose(ee_pose_goal, ee_pose_inv))
        # torch.set_printoptions(precision=2, sci_mode=False)
        # print(error[..., -1, :])

        # C, dC/dx -- Task space error
        cost = torch.linalg.norm(error, dim=-1)
        gradient_cost_wrt_x = -1.0 * error  # multiply by -1 because we return the gradient of the cost wrt W_EE_current

        # (dx/dq)^T @ dC/dx (jacobian transpose x task space error)
        # sum the gradient and cost over the task space links
        grad_cost_wrt_q_pos = torch.einsum("...dj,...d->...j", jacs_spatial_th_ee.squeeze(-3), gradient_cost_wrt_x)

        grad_cost_wrt_cp = self.compute_grad_cost_wrt_cp(
            control_points,
            grad_cost_wrt_q_pos,
            get_type_single="pos",
        )

        # We only want to adjust the last control point of the trajectory, so the gradient of all others is set to zero
        grad_cost_wrt_cp[..., :-1, :] = 0.0
        # The cost is the pose error at the last trajectory point, so we also set the cost of all other points to zero
        cost[..., :-1] = 0.0

        return cost, grad_cost_wrt_cp
