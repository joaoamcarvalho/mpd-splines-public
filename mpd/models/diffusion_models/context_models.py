import torch
import torch.nn as nn

from mpd.models.layers.layers import MLP


class ContextModelEEPoseGoal(nn.Module):

    def __init__(self, out_dim=64, n_layers=2, act="relu", **kwargs):
        super().__init__()
        # 9d representation of rotation
        # 3d representation of position
        self.in_dim = 9 + 3
        self.out_dim = out_dim

        # self.net = nn.Identity()
        self.net = MLP(self.in_dim, out_dim, hidden_dim=out_dim, n_layers=n_layers, act=act)

    def forward(self, ee_goal_orientation_normalized, ee_goal_position_normalized, **kwargs):
        pose_repr = torch.cat((ee_goal_orientation_normalized, ee_goal_position_normalized), dim=-1)
        emb = self.net(pose_repr)
        return emb


class ContextModelQs(nn.Module):

    def __init__(self, in_dim, out_dim=64, n_layers=2, act="relu", **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # self.net = nn.Identity()
        self.net = MLP(self.in_dim, out_dim, hidden_dim=out_dim, n_layers=n_layers, act=act)

    def forward(self, qs_normalized=None, **kwargs):
        emb = self.net(qs_normalized)
        return emb


class ContextModelCombined(nn.Module):

    def __init__(
        self, context_model_qs=None, context_model_ee_pose_goal=None, out_dim=64, n_layers=1, act="relu", **kwargs
    ):
        assert not (context_model_qs is None and context_model_ee_pose_goal is None)
        super().__init__()

        self.context_model_qs = context_model_qs
        self.context_model_ee_pose_goal = context_model_ee_pose_goal

        self.in_dim = 0
        if self.context_model_qs is not None:
            self.in_dim += self.context_model_qs.out_dim
        if self.context_model_ee_pose_goal is not None:
            self.in_dim += self.context_model_ee_pose_goal.out_dim

        self.out_dim = out_dim
        self.net = MLP(self.in_dim, self.out_dim, hidden_dim=out_dim, n_layers=n_layers, act=act)

    def forward(
        self, qs_normalized=None, ee_goal_orientation_normalized=None, ee_goal_position_normalized=None, **kwargs
    ):
        emb_q = None
        if self.context_model_qs is not None:
            emb_q = self.context_model_qs(qs_normalized)

        emb_ee_goal_pose = None
        if self.context_model_ee_pose_goal is not None:
            emb_ee_goal_pose = self.context_model_ee_pose_goal(
                ee_goal_orientation_normalized, ee_goal_position_normalized
            )

        if emb_q is not None and emb_ee_goal_pose is not None:
            emb = torch.cat((emb_q, emb_ee_goal_pose), dim=-1)
        elif emb_q is not None:
            emb = emb_q
        elif emb_ee_goal_pose is not None:
            emb = emb_ee_goal_pose

        context_emb = self.net(emb)
        return context_emb
