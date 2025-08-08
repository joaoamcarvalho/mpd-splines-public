import abc
from typing import Tuple

from mpd.parametric_trajectory.phase_time import PhaseTimeLinear, PhaseTimeSigmoid
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS


class ParametricTrajectoryBase(abc.ABC):

    def __init__(
        self,
        num_T_pts=128,
        trajectory_duration=5.0,
        tensor_args=DEFAULT_TENSOR_ARGS,
        phase_time_class="PhaseTimeLinear",
        phase_time_args={},
        **kwargs,
    ):
        self.num_T_pts = num_T_pts
        self.trajectory_duration = trajectory_duration
        self.tensor_args = tensor_args

        self.q_pos_start = None
        self.q_pos_goal = None

        phase_time_class_fn_d = {"PhaseTimeLinear": PhaseTimeLinear, "PhaseTimeSigmoid": PhaseTimeSigmoid}
        assert (
            phase_time_class in phase_time_class_fn_d
        ), f"phase_time_class must be one of {list(phase_time_class_fn_d.keys())}"

        self.phase_time = phase_time_class_fn_d[phase_time_class](
            trajectory_duration=trajectory_duration, num_T_pts=num_T_pts, **phase_time_args, tensor_args=tensor_args
        )

    def get_timesteps(self, *args, **kwargs):
        return self.phase_time.t

    def get_phase_steps(self, *args, **kwargs):
        return self.phase_time.s

    def get_q_pos_start_q_goal(self, q_pos_start, q_pos_goal):
        q_pos_start = self.q_pos_start if q_pos_start is None else q_pos_start
        q_pos_goal = self.q_pos_goal if q_pos_goal is None else q_pos_goal
        return q_pos_start, q_pos_goal

    @abc.abstractmethod
    def augment_control_points_fn(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def preprocess_control_points(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def get_q_trajectory_in_phase(self, *args, **kwargs):
        raise NotImplementedError

    def get_q_trajectory(
        self,
        q_control_points,
        q_pos_start,
        q_pos_goal,
        get_type: Tuple = ("pos", "vel", "acc"),
        get_time_representation=True,  # if False, returns the trajectory in phase
        **kwargs,
    ):
        q_pos_start, q_pos_goal = self.get_q_pos_start_q_goal(q_pos_start, q_pos_goal)

        q_control_points = self.augment_control_points_fn(q_control_points, q_pos_start, q_pos_goal)
        q_control_points = self.preprocess_control_points(q_control_points)
        if q_control_points.ndim < 2:
            raise NotImplementedError

        q_traj_in_phase_d = self.get_q_trajectory_in_phase(q_control_points, get_type=get_type)
        q_traj_d = {}
        if "pos" in get_type:
            q_traj_d["pos"] = q_traj_in_phase_d["pos"]
        if "vel" in get_type:
            if get_time_representation:
                q_traj_d["vel"] = q_traj_in_phase_d["vel"] * self.phase_time.rs[..., None]
            else:
                q_traj_d["vel"] = q_traj_in_phase_d["vel"]
        if "acc" in get_type:
            if get_time_representation:
                q_traj_d["acc"] = (
                    q_traj_in_phase_d["acc"] * self.phase_time.rs[..., None] ** 2
                    + q_traj_in_phase_d["vel"] * self.phase_time.dr_ds[..., None] * self.phase_time.rs[..., None]
                )
            else:
                q_traj_d["acc"] = q_traj_in_phase_d["acc"]

        return q_traj_d
