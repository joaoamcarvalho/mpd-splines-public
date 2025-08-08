import abc

import torch

from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS


def convert_r_to_t(r):
    # integrates r(s) to get times t
    # r = ds/dt
    # ds = r.shape[0]
    dt = 1.0 / r / r.shape[0]  # dt = 1/(ds/dt) * ds
    # t = integral(dt) = cumsum(dt)
    ts = torch.cumsum(torch.cat([torch.zeros_like(dt[0]).view(-1), dt[:-1]], dim=-1), dim=-1)
    return ts


class PhaseTime(abc.ABC):

    def __init__(self, trajectory_duration=5.0, num_T_pts=128, tensor_args=DEFAULT_TENSOR_ARGS, **kwargs):
        self.trajectory_duration = trajectory_duration
        self.num_T_pts = num_T_pts

        self.tensor_args = tensor_args

        # phase variable
        self.s = torch.linspace(0, 1, num_T_pts, **tensor_args, requires_grad=True)
        self.ds = 1.0 / num_T_pts

        # adjust r(s) to match the desired trajectory duration
        rs_tmp = self.rs_fn(self.s)
        t_tmp = convert_r_to_t(rs_tmp)
        ratio = t_tmp[-1] / self.trajectory_duration

        # r(s) = ds/dt
        self.rs = rs_tmp * ratio
        # r(s)^-1 = dt/ds
        self.rs_inv = 1.0 / self.rs

        # dr(s)/ds
        self.dr_ds = self.drs_ds_fn(self.s) * ratio

        # time variable
        self.t = convert_r_to_t(self.rs)

    @abc.abstractmethod
    def rs_fn(self, s):
        # implements r(s) = ds/dt as a function of s in [0, 1]
        raise NotImplementedError

    @abc.abstractmethod
    def drs_ds_fn(self, s):
        # implements dr(s)/ds as a function of s in [0, 1]
        raise NotImplementedError

    def phi_s(self, s):
        # s in [0, 1]
        raise NotImplementedError

    def phi_inv_t(self, t):
        # t in [0, T]
        raise NotImplementedError


class PhaseTimeLinear(PhaseTime):
    # Phase and time evolving linearly with t = phi(s) = s * T
    # where s in [0, 1] is the linearly evolving phase, and T is the trajectory duration

    def rs_fn(self, s):
        return torch.ones_like(s) / self.trajectory_duration

    def drs_ds_fn(self, s):
        return torch.zeros_like(s)

    def phi_s(self, s):
        return s * self.trajectory_duration

    def phi_inv_t(self, t):
        return t / self.trajectory_duration


class PhaseTimeSigmoid(PhaseTime):
    # "Sigmoid" relation between phase and time
    # Prevents the velocity and acceleration to abruptly change at the start and end of the trajectory

    def __init__(self, alpha=2.1, **kwargs):
        assert alpha > 2, "alpha must be greater than 2"
        self.alpha = alpha
        self.eps = 1.0
        super().__init__(**kwargs)

    def rs_fn(self, s):
        return 4**self.alpha * s ** (self.alpha - 1) * (1 - s) ** (self.alpha - 1) + self.eps

    def drs_ds_fn(self, s):
        return (
            4**self.alpha
            * (self.alpha - 1)
            * (
                s ** (self.alpha - 2) * (1 - s) ** (self.alpha - 1)
                - s ** (self.alpha - 1) * (1 - s) ** (self.alpha - 2)
            )
        )
