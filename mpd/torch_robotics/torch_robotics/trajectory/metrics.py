import torch
from torchlie.functional.se3_impl import _fns as SE3_Func
from vendi_score import vendi


def compute_path_length(trajs, robot):
    assert trajs.ndim == 3  # batch, horizon, state_dim
    trajs_pos = robot.get_position(trajs)
    path_length = torch.linalg.norm(torch.diff(trajs_pos, dim=-2), dim=-1).sum(-1)
    return path_length


def compute_smoothness(trajs, robot, trajs_acc=None):
    if trajs_acc is None:
        assert trajs.ndim == 3
        trajs_acc = robot.get_acceleration(trajs)
    else:
        assert trajs_acc.ndim == 3
    smoothness = torch.linalg.norm(trajs_acc, dim=-1)
    smoothness = smoothness.sum(-1)  # sum over the trajectory horizon
    return smoothness


def compute_trajectory_diversity(trajs, robot, method="vendi_score"):
    assert trajs.ndim == 3  # batch, horizon, state_dim
    trajs_pos = robot.get_position(trajs)

    if method == "waypoint_variance":
        sum_var_waypoints = 0.0
        for via_points in trajs_pos.permute(1, 0, 2):  # horizon, batch, position
            parwise_distance_between_waypoints = torch.cdist(via_points, via_points, p=2)
            distances = torch.triu(parwise_distance_between_waypoints, diagonal=1).view(-1)
            sum_var_waypoints += torch.var(distances)
        return sum_var_waypoints
    elif method == "vendi_score":
        kernel = lambda x, y: torch.exp(-torch.linalg.norm(x - y) ** 2)
        return vendi.score(trajs_pos, kernel)
    else:
        raise NotImplementedError(f"Method {method} not implemented")


def compute_ee_pose_errors(ee_pose_goal, ee_pose_goal_achieved):
    ee_pose_goal_achieved_inv = SE3_Func.inv(ee_pose_goal_achieved[..., :3, :4])
    error_ee_pose_goal = SE3_Func.log(
        SE3_Func.compose(ee_pose_goal[..., :3, :4], ee_pose_goal_achieved_inv[..., :3, :4])
    )
    error_ee_pose_goal_position = error_ee_pose_goal[..., :3]
    error_ee_pose_goal_orientation = error_ee_pose_goal[..., 3:]
    return error_ee_pose_goal_position, error_ee_pose_goal_orientation
