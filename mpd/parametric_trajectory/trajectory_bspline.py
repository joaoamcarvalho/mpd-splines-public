import isaacgym

from functools import partial
from typing import Tuple


import einops
import numpy as np
import torch

from matplotlib import pyplot as plt

from mpd.parametric_trajectory.trajectory_base import ParametricTrajectoryBase
from mpd.parametric_trajectory.phase_time import PhaseTimeLinear, PhaseTimeSigmoid
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.plot_utils import create_fig_and_axes


class ParametricTrajectoryBspline(ParametricTrajectoryBase):

    def __init__(
        self,
        n_control_points=18,
        degree=5,
        zero_vel_at_start_and_goal=True,
        zero_acc_at_start_and_goal=True,
        remove_outer_control_points=False,
        keep_last_control_point=False,
        num_T_pts=128,
        trajectory_duration=5.0,
        phase_time_class="PhaseTimeLinear",
        phase_time_args={},
        tensor_args=DEFAULT_TENSOR_ARGS,
        **kwargs,
    ):
        super().__init__(
            num_T_pts=num_T_pts,
            trajectory_duration=trajectory_duration,
            tensor_args=tensor_args,
            phase_time_class=phase_time_class,
            phase_time_args=phase_time_args,
        )

        self.bspline = BSpline(num_pts=n_control_points, degree=degree, num_T_pts=num_T_pts, **tensor_args)
        self.bspline_basis_map = {"pos": self.bspline.N, "vel": self.bspline.dN, "acc": self.bspline.ddN}
        self.zero_vel_at_start_and_goal = zero_vel_at_start_and_goal
        self.zero_acc_at_start_and_goal = zero_acc_at_start_and_goal
        self.remove_outer_control_points = remove_outer_control_points
        self.keep_last_control_point = keep_last_control_point

    def remove_control_points_fn(self, cps):
        # Remove the first and last control points in position, velocity and acceleration
        # keep the last control point if we condition on the ee pose goal
        last_control_point = cps[..., -1, :].clone()
        if self.remove_outer_control_points:
            cps = cps[..., 1:-1, :]
            if self.zero_vel_at_start_and_goal:
                cps = cps[..., 1:-1, :]
            if self.zero_acc_at_start_and_goal:
                cps = cps[..., 1:-1, :]
            if self.keep_last_control_point:
                cps = torch.cat((cps, last_control_point[..., None, :]), dim=-2)
        return cps

    def augment_control_points_fn(self, control_points, q_pos_start=None, q_pos_goal=None):
        q_pos_start, q_pos_goal = self.get_q_pos_start_q_goal(q_pos_start, q_pos_goal)

        if control_points.ndim - q_pos_start.ndim == 1:
            q_pos_start = einops.rearrange(q_pos_start, "... d -> ... 1 d")
            q_pos_goal = einops.rearrange(q_pos_goal, "... d -> ... 1 d")
        else:
            repeat_shape = control_points.shape[:-2]
            if len(repeat_shape) == 1:
                q_pos_start = einops.repeat(q_pos_start, "... -> n 1 ...", n=repeat_shape[0])
                q_pos_goal = einops.repeat(q_pos_goal, "... -> n 1 ...", n=repeat_shape[0])
            elif len(repeat_shape) == 2:
                q_pos_start = einops.repeat(q_pos_start, "... -> n m 1 ...", n=repeat_shape[0], m=repeat_shape[1])
                q_pos_goal = einops.repeat(q_pos_goal, "... -> n m 1 ...", n=repeat_shape[0], m=repeat_shape[1])

        last_inner_control_point = control_points[..., -1, :].clone()[..., None, :]
        if self.keep_last_control_point:
            control_points = control_points[..., :-1, :]
        control_points_augmented = control_points.clone()
        if self.remove_outer_control_points:
            # add control points in between the initial and goal states
            if self.zero_acc_at_start_and_goal:
                if self.keep_last_control_point:
                    control_points_augmented = torch.cat(
                        [q_pos_start, control_points_augmented, last_inner_control_point], dim=-2
                    )
                else:
                    control_points_augmented = torch.cat([q_pos_start, control_points_augmented, q_pos_goal], dim=-2)

            if self.zero_vel_at_start_and_goal:
                if self.keep_last_control_point:
                    control_points_augmented = torch.cat(
                        [q_pos_start, control_points_augmented, last_inner_control_point], dim=-2
                    )
                else:
                    control_points_augmented = torch.cat([q_pos_start, control_points_augmented, q_pos_goal], dim=-2)

            if self.keep_last_control_point:
                control_points_augmented = torch.cat(
                    [q_pos_start, control_points_augmented, last_inner_control_point], dim=-2
                )
            else:
                control_points_augmented = torch.cat([q_pos_start, control_points_augmented, q_pos_goal], dim=-2)

        return control_points_augmented

    def preprocess_control_points(self, q_control_points):
        # ensure b-spline boundary conditions
        if self.zero_vel_at_start_and_goal:
            q_control_points[..., 1, :] = q_control_points[..., 0, :]
            q_control_points[..., -2, :] = q_control_points[..., -1, :]
        if self.zero_acc_at_start_and_goal:
            q_control_points[..., 2, :] = q_control_points[..., 0, :]
            q_control_points[..., -3, :] = q_control_points[..., -1, :]
        return q_control_points

    def get_q_trajectory_in_phase(self, q_control_points: torch.Tensor, get_type: Tuple = ("pos", "vel", "acc")):
        # Bspline interpolation
        q_in_phase_d = {}
        for t in get_type:
            assert t in self.bspline_basis_map, f"get_type element must be one of {list(self.bspline_basis_map.keys())}"
            if t in self.bspline_basis_map:
                q_in_phase_d[t] = torch.einsum("ijk,...km->...jm", self.bspline_basis_map[t], q_control_points)
        return q_in_phase_d

    def get_grad_q_traj_in_phase_wrt_control_points(
        self, *args, get_type: Tuple = ("pos", "vel", "acc"), remove_control_points=False, **kwargs
    ):
        # q(s) = B(s) @ control_points
        # dq(s)/dcontrol_points = B(s)
        grad_d = {}
        for t in get_type:
            assert t in self.bspline_basis_map, f"get_type element must be one of {list(self.bspline_basis_map.keys())}"
            M = self.bspline_basis_map[t]
            if remove_control_points:
                # If the outer control points are fixed, we can remove them to compute gradients only
                # wrt the inner control points
                M = self.remove_control_points_fn(M.unsqueeze(-1)).squeeze(-1)
            grad_d[t] = M
        return grad_d

    def get_control_points_derivatives(
        self, inner_control_points, q_pos_start=None, q_pos_goal=None, get_type="pos", get_time_repr=True, **kwargs
    ):
        raise NotImplementedError
        # https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-derv.html
        # https://arxiv.org/pdf/2405.01758 - appendix
        q_pos_start, q_pos_goal = self.get_q_pos_start_q_goal(q_pos_start, q_pos_goal)
        all_control_points = self.augment_control_points_fn(inner_control_points, q_pos_start, q_pos_goal)
        all_control_points = self.preprocess_control_points(all_control_points)

        if get_type == "pos":
            return all_control_points
        elif get_type == "vel":
            c_i = all_control_points[..., :-1, :]
            c_i_1 = all_control_points[..., 1:, :]
            control_points_vel = self.bspline.coeff_control_points_vel[..., None] * (c_i_1 - c_i)
            if get_time_repr:
                assert isinstance(self.phase_time, PhaseTimeLinear), "Only linear phase time is supported"
                ds_dt = self.phase_time.rs[0]  # for linear phase all values are the same
                control_points_vel = control_points_vel * (ds_dt**-1)
            return control_points_vel
        elif get_type == "acc":
            # TODO - remove recomputation of control_points_vel
            # Recompute control points for the velocity bspline
            c_i = all_control_points[..., :-1, :]
            c_i_1 = all_control_points[..., 1:, :]
            control_points_vel = self.bspline.coeff_control_points_vel[..., None] * (c_i_1 - c_i)
            control_points_vel_i = control_points_vel[..., :-1, :]
            control_points_vel_i_1 = control_points_vel[..., 1:, :]
            control_points_acc = self.bspline.coeff_control_points_acc[..., None] * (
                control_points_vel_i_1 - control_points_vel_i
            )
            if get_time_repr:
                assert isinstance(self.phase_time, PhaseTimeLinear), "Only linear phase time is supported"
                ds_dt = self.phase_time.rs[0]  # for linear phase all values are the same
                control_points_acc = control_points_acc * (ds_dt**-2)
            return control_points_acc
        elif get_type == "jerk":
            pass
        else:
            raise NotImplementedError


class BSpline:
    def __init__(self, num_pts, degree=7, num_T_pts=1024, device="cpu", dtype=torch.float32, **kwargs):
        self.num_T_pts = num_T_pts
        self.d = degree
        self.n_pts = num_pts
        self.m = self.d + self.n_pts
        self.u = np.pad(np.linspace(0.0, 1.0, self.m + 1 - 2 * self.d), self.d, "edge")
        self.N, self.dN, self.ddN, self.dddN = self.calculate_N()
        self.N = torch.from_numpy(self.N).to(dtype).to(device)
        self.dN = torch.from_numpy(self.dN).to(dtype).to(device)
        self.ddN = torch.from_numpy(self.ddN).to(dtype).to(device)
        self.dddN = torch.from_numpy(self.dddN).to(dtype).to(device)

        # Coefficients for bspline derivatives
        # https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-derv.html
        # https://arxiv.org/pdf/2405.01758 - appendix
        # V_i = coeff * (P_i+1 - P_i)
        u_i_1 = self.u[1:num_pts]
        u_i_p_1 = self.u[1 + degree : num_pts + degree]
        self.coeff_control_points_vel = torch.tensor(degree / (u_i_p_1 - u_i_1), dtype=dtype, device=device)

        # A_i = coeff * (V_i+1 - V_i)
        u_i_2 = self.u[2:num_pts]
        u_i_p_2 = self.u[2 + degree : num_pts + degree]
        self.coeff_control_points_acc = torch.tensor((degree - 1) / (u_i_p_2 - u_i_2), dtype=dtype, device=device)

    def calculate_N(self):
        def N(n, t, i):
            if n == 0:
                if self.u[i] <= t < self.u[i + 1]:
                    return 1
                else:
                    return 0
            s = 0.0
            if self.u[i + n] - self.u[i] != 0:
                s += (t - self.u[i]) / (self.u[i + n] - self.u[i]) * N(n - 1, t, i)
            if self.u[i + n + 1] - self.u[i + 1] != 0:
                s += (self.u[i + n + 1] - t) / (self.u[i + n + 1] - self.u[i + 1]) * N(n - 1, t, i + 1)
            return s

        def dN(n, t, i):
            m1 = self.u[i + n] - self.u[i]
            m2 = self.u[i + n + 1] - self.u[i + 1]
            s = 0.0
            if m1 != 0:
                s += N(n - 1, t, i) / m1
            if m2 != 0:
                s -= N(n - 1, t, i + 1) / m2
            return n * s

        def ddN(n, t, i):
            m1 = self.u[i + n] - self.u[i]
            m2 = self.u[i + n + 1] - self.u[i + 1]
            s = 0.0
            if m1 != 0:
                s += dN(n - 1, t, i) / m1
            if m2 != 0:
                s -= dN(n - 1, t, i + 1) / m2
            return n * s

        def dddN(n, t, i):
            m1 = self.u[i + n] - self.u[i]
            m2 = self.u[i + n + 1] - self.u[i + 1]
            s = 0.0
            if m1 != 0:
                s += ddN(n - 1, t, i) / m1
            if m2 != 0:
                s -= ddN(n - 1, t, i + 1) / m2
            return n * s

        T = np.linspace(0.0, 1.0, self.num_T_pts)
        Ns = [np.stack([N(self.d, t, i) for i in range(self.m - self.d)]) for t in T]
        Ns = np.stack(Ns, axis=0)
        Ns[-1, -1] = 1.0
        dNs = [np.stack([dN(self.d, t, i) for i in range(self.m - self.d)]) for t in T]
        dNs = np.stack(dNs, axis=0)
        dNs[-1, -1] = (self.m - 2 * self.d) * self.d
        dNs[-1, -2] = -(self.m - 2 * self.d) * self.d
        ddNs = [np.stack([ddN(self.d, t, i) for i in range(self.m - self.d)]) for t in T]
        ddNs = np.stack(ddNs, axis=0)
        ddNs[-1, -1] = 2 * self.d * (self.m - 2 * self.d) ** 2 * (self.d - 1) / 2
        ddNs[-1, -2] = -3 * self.d * (self.m - 2 * self.d) ** 2 * (self.d - 1) / 2
        ddNs[-1, -3] = self.d * (self.m - 2 * self.d) ** 2 * (self.d - 1) / 2
        dddNs = [np.stack([dddN(self.d, t, i) for i in range(self.m - self.d)]) for t in T]
        dddNs = np.stack(dddNs, axis=0)
        dddNs[-1, -1] = 6 * self.d * (self.m - 2 * self.d) ** 3 * (self.d - 2)
        dddNs[-1, -2] = -10.5 * self.d * (self.m - 2 * self.d) ** 3 * (self.d - 2)
        dddNs[-1, -3] = 5.5 * self.d * (self.m - 2 * self.d) ** 3 * (self.d - 2)
        dddNs[-1, -4] = -self.d * (self.m - 2 * self.d) ** 3 * (self.d - 2)
        return Ns[np.newaxis], dNs[np.newaxis], ddNs[np.newaxis], dddNs[np.newaxis]


if __name__ == "__main__":

    n_control_points = 16
    control_points_x = np.linspace(-1, 1, n_control_points, dtype=np.float32)
    control_points_y = control_points_x + 0.2 * np.sin(control_points_x * 10.0)
    control_points = np.stack((control_points_x, control_points_y), axis=1)

    num_T_pts = 128
    trajectory_duration = 5.0

    fig_cps, axs_cps = create_fig_and_axes(2, figsize=(8, 6))
    fig_qs, axs_qs = plt.subplots(control_points.shape[-1], 3, figsize=(12, 8))
    fig_phase, axs_phase = create_fig_and_axes(2, figsize=(8, 6))
    fig_rs, axs_rs = create_fig_and_axes(2, figsize=(8, 6))
    for l, (phase_class, phase_time_args, line_color, line_style) in enumerate(
        zip(["PhaseTimeLinear", "PhaseTimeSigmoid"], [{}, {"k": 8}], ["gray", "orange"], ["solid", "solid"])
    ):
        print(f"Phase class: {phase_class}")

        parametric_traj = ParametricTrajectoryBspline(
            n_control_points=n_control_points,
            degree=5,
            num_T_pts=num_T_pts,
            zero_vel_at_start_and_goal=True,
            zero_acc_at_start_and_goal=True,
            trajectory_duration=trajectory_duration,
            phase_time_class=phase_class,
            phase_time_args=phase_time_args,
            tensor_args=dict(device="cpu", dtype=torch.float32),
        )

        control_points_th = torch.from_numpy(control_points)[None, ...]
        q_start = control_points_th[0, 0, ...]
        q_goal = control_points_th[0, -1, ...]

        q_traj_d = parametric_traj.get_q_trajectory(
            control_points_th, q_start, q_goal, get_type=("pos", "vel", "acc"), get_time_representation=True
        )

        q = q_traj_d["pos"][0].detach()
        dq = q_traj_d["vel"][0].detach()
        ddq = q_traj_d["acc"][0].detach()

        # plot control points and q_pos
        axs_cps.scatter(control_points_x, control_points_y, c="blue", marker="o", s=10**2, zorder=100)
        axs_cps.plot(q[:, 0], q[:, 1], color=line_color, linestyle=line_style, linewidth=8)

        # plot q, dq, ddq in time
        t = parametric_traj.phase_time.t

        # integrate q and dq for checking correctness
        print(f"q(T) (from B-spline): {q[-1]}")
        q_T_from_dq = (
            q[0] + torch.cumsum(dq * torch.diff(torch.cat([torch.zeros([1]), t]), dim=-1)[..., None], dim=0)[-1]
        )
        print(f"q(T) (from dq integration): {q_T_from_dq}")

        print(f"dq(T) (from B-spline): {dq[-1]}")
        dq_T_from_ddq = (
            dq[0] + torch.cumsum(ddq * torch.diff(torch.cat([torch.zeros([1]), t]), dim=-1)[..., None], dim=0)[-1]
        )
        print(f"dq(T) (from ddq integration): {dq_T_from_ddq}\n")

        for i, x in enumerate([q, dq, ddq]):
            for j in range(x.shape[1]):
                axs_qs[j, i].plot(t.detach().cpu(), x[:, j], linestyle=line_style, color=line_color, linewidth=4)
                axs_qs[j, i].set_yticks([x[:, j].min(), x[:, j].max()])
                axs_qs[j, i].set_yticklabels([f"${x[:, j].min():.2f}$", f"${x[:, j].max():.2f}$"])

        if phase_class == "PhaseTimeLinear":
            phase_time_class_fn = PhaseTimeLinear
        elif phase_class == "PhaseTimeSigmoid":
            phase_time_class_fn = partial(PhaseTimeSigmoid, **phase_time_args)
        else:
            raise NotImplementedError

        # plot the phase variable in time
        phase = phase_time_class_fn(trajectory_duration=trajectory_duration, num_T_pts=num_T_pts)
        t_np = phase.t.cpu().detach().numpy()
        axs_phase.plot(t_np, phase.s.cpu().detach(), color=line_color, linestyle="solid", linewidth=4, label="s")
        axs_phase.plot(t_np, phase.rs.cpu().detach(), color=line_color, linestyle="dashed", linewidth=4, label="ds/dt")
        axs_phase.plot(
            t_np, phase.dr_ds.cpu().detach(), color=line_color, linestyle="dotted", linewidth=4, label="d^2s/dt^2"
        )
        axs_phase.set_xlabel("t")
        axs_phase.legend()

        # plot ds_dt in phase
        s_np = phase.s.cpu().detach().numpy()
        axs_rs.plot(s_np, phase.rs.cpu().detach(), color=line_color, linestyle="dashed", linewidth=4, label="ds/dt")
        axs_rs.plot(
            s_np, phase.dr_ds.cpu().detach(), color=line_color, linestyle="dotted", linewidth=4, label="d^2s/dt^2"
        )
        axs_rs.set_xlabel("s")
        axs_rs.legend()

    plt.show()
