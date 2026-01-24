#!/usr/bin/env python3
"""
Standalone Collision Model wrapper.

- 只依赖 torch / numpy / scipy（公共库）。
- 内置 ResMLP 和 DataProcessor 的逻辑，并将均值/方差常量写死。
- 接口:
    1) predict_raw(inputs_raw): 输入 (N,14) -> 输出 (N,6)，均为未归一化物理量。
    2) predict_post_velocity_2d(pre_v_2d, yaw_rad, friction, damping): 单样本高层接口，输出 2D 反弹速度。

使用示例:
  from collision_model_standalone import CollisionPredictor

  predictor = CollisionPredictor(ckpt_path="Mujoco/Models_MultiMat/best_model_multimat.pth")

  v_post_2d = predictor.predict_post_velocity_2d(
      pre_v_2d=[2.0, -1.0],
      yaw_rad=0.1,
      friction=0.5,
      damping=0.2,
  )
"""

import math
import os
from pathlib import Path
from typing import Sequence, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R


# ================= 固定统计量（你提供的均值和方差） =================

STATE_MEAN = np.array(
    [4.8891096115112305, 0.00619716988876462, 0.25364741683006287, 0.0008679787279106677, 1.1658222675323486, 0.0009014693205244839],
    dtype=np.float32,
)
STATE_STD = np.array(
    [1.148665428161621, 2.3237507343292236, 0.2325383871793747, 0.6568835377693176, 1.1716166734695435, 0.22680452466011047],
    dtype=np.float32,
)
PHYS_MEAN = np.array([0.499712198972702, 0.2758626639842987], dtype=np.float32)
PHYS_STD = np.array([0.2307351529598236, 0.1295028179883957], dtype=np.float32)
TARGET_MEAN = np.array(
    [-1.3977482318878174, -0.001275352668017149, -0.6846954822540283, 0.0035870324354618788, 8.036012649536133, -0.015603979118168354],
    dtype=np.float32,
)
TARGET_STD = np.array(
    [0.4898343086242676, 1.476805567741394, 0.5955789089202881, 4.446241855621338, 4.452792167663574, 5.49974250793457],
    dtype=np.float32,
)


# ================= 模型结构 (ResidualBlock + ResMLP) =================

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class ResMLP(nn.Module):
    def __init__(self, input_dim: int = 14, output_dim: int = 6, hidden_dim: int = 1024, num_layers: int = 6):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_layers)])
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        for block in self.res_blocks:
            x = block(x)
        return self.decoder(x)


# ================= Processor: 使用写死的均值/方差 =================

class CollisionProcessor:
    """
    复刻 DataProcessor 的 normalize_input / denormalize_target，但使用写死的常量。
    """

    def __init__(self, device: Optional[torch.device] = None):
        if device is None:
            device = torch.device("cpu")
        self.device = device

        self.state_mean = torch.from_numpy(STATE_MEAN).to(device=device)
        self.state_std = torch.from_numpy(STATE_STD).to(device=device)
        self.phys_mean = torch.from_numpy(PHYS_MEAN).to(device=device)
        self.phys_std = torch.from_numpy(PHYS_STD).to(device=device)
        self.target_mean = torch.from_numpy(TARGET_MEAN).to(device=device)
        self.target_std = torch.from_numpy(TARGET_STD).to(device=device)

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N,14) = [state(6), rot6d(6), phys(2)]
        返回: 归一化后的 (N,14)
        """
        x = x.to(self.device)
        state = x[:, :6]
        rot = x[:, 6:-2]
        phys = x[:, -2:]

        state_norm = (state - self.state_mean) / self.state_std
        phys_norm = (phys - self.phys_mean) / self.phys_std

        return torch.cat([state_norm, rot, phys_norm], dim=1)

    def denormalize_target(self, y_norm: torch.Tensor) -> torch.Tensor:
        """
        y_norm: (N,6) 归一化输出 -> 未归一化目标 (N,6)
        """
        y_norm = y_norm.to(self.device)
        return y_norm * self.target_std + self.target_mean


# ================= 辅助: 四元数 -> 6D 旋转 =================

def quat_to_6d(quat_tensor: torch.Tensor) -> torch.Tensor:
    """
    将四元数 (w, x, y, z) 转为 6D 旋转 (R 的前两列 flatten)。
    输入: quat_tensor shape (N,4) in (w,x,y,z).
    返回: (N,6) float32 tensor.
    """
    q = quat_tensor.detach().cpu().numpy()
    q_scipy = np.concatenate([q[:, 1:], q[:, 0:1]], axis=1)  # (x,y,z,w)
    r = R.from_quat(q_scipy)
    matrix = r.as_matrix()  # (N,3,3)
    r6d = np.concatenate([matrix[:, :, 0], matrix[:, :, 1]], axis=1)  # (N,6)
    return torch.tensor(r6d, dtype=torch.float32)


# ================= 高层封装: CollisionPredictor =================

class CollisionPredictor:
    """
    用法:
      predictor = CollisionPredictor("Mujoco/Models_MultiMat/best_model_multimat.pth")

      # 低层接口：直接给原生 inputs_raw (N,14)
      outputs = predictor.predict_raw(inputs_raw)  # -> (N,6)

      # 高层接口：从 pre_v_2d + yaw + friction + damping 得到 post_v_2d
      v_post_2d = predictor.predict_post_velocity_2d(
          pre_v_2d=[2.0, 0.0],
          yaw_rad=0.1,
          friction=0.5,
          damping=0.3,
      )
    """

    def __init__(self, ckpt_path: Optional[str] = None, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if ckpt_path is None:
            # 默认相对路径与原仓库一致
            ckpt_path = "small/best_model_multimat.pth"
        self.ckpt_path = ckpt_path

        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"Collision model checkpoint not found at: {self.ckpt_path}")

        self.model = ResMLP(input_dim=14, output_dim=6, hidden_dim=1024, num_layers=6).to(self.device)
        state = torch.load(self.ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            # 万一被包在 dict 里
            state = state["model"]
        self.model.load_state_dict(state)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.processor = CollisionProcessor(device=self.device)

    @torch.no_grad()
    def predict_raw(self, inputs_raw: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        输入:
          inputs_raw: shape (N,14)，未归一化:
            [v(3), w(3), rot6d(6), params(2)]
        输出:
          outputs_phys: shape (N,6)，未归一化物理量 (post_v(3), post_w(3))
        """
        if not torch.is_tensor(inputs_raw):
            x = torch.from_numpy(np.asarray(inputs_raw, dtype=np.float32))
        else:
            x = inputs_raw.float()
        x = x.to(self.device)

        x_norm = self.processor.normalize_input(x)   # (N,14)
        y_norm = self.model(x_norm)                  # (N,6)
        y_phys = self.processor.denormalize_target(y_norm)  # (N,6)
        return y_phys.detach().cpu().numpy()

    @torch.no_grad()
    def predict_post_velocity_2d(
        self,
        pre_v_2d: Sequence[float],
        yaw_rad: float,
        friction: float,
        damping: float,
    ) -> np.ndarray:
        """
        高层接口：给碰撞前 2D 速度 + yaw + 墙参数，返回 2D 反弹速度。

        pre_v_2d: [vx, vy] (world frame velocity at collision).
        yaw_rad: float yaw angle (rad), 旋转轴为 z.
        friction, damping: 标量。

        返回:
          post_v_2d: np.ndarray shape (2,)
        """
        pre_v_2d = np.asarray(pre_v_2d, dtype=np.float32).reshape(2)
        # 扩展为 3D v (z 分量 0)
        pre_v_3d = np.concatenate([pre_v_2d, [0.0]], axis=0)
        pre_v = torch.from_numpy(pre_v_3d[None, :]).float()  # (1,3)

        pre_w = torch.zeros(1, 3, dtype=torch.float32)       # (1,3)

        # yaw -> 四元数 (w,x,y,z)，绕 z 轴
        th = float(yaw_rad)
        w_q = math.cos(th * 0.5)
        z_q = math.sin(th * 0.5)
        pre_q = torch.tensor([[w_q, 0.0, 0.0, z_q]], dtype=torch.float32)  # (1,4)
        pre_rot6d = quat_to_6d(pre_q)                                     # (1,6)

        phys_params = torch.tensor([[float(friction), float(damping)]], dtype=torch.float32)  # (1,2)

        inputs_raw = torch.cat([pre_v, pre_w, pre_rot6d, phys_params], dim=1)  # (1,14)

        y_phys = self.predict_raw(inputs_raw)  # (1,6)
        post_v_3d = y_phys[0, :3]             # (3,)
        post_v_2d = post_v_3d[:2]             # (2,)
        return post_v_2d.astype(np.float32)


# ================= CLI Demo =================

def _demo():
    predictor = CollisionPredictor()

    pre_v_2d = [1.27, 1.14]
    yaw_rad = 0.0
    friction = 0.65
    damping = 0.1

    post_v_2d = predictor.predict_post_velocity_2d(pre_v_2d, yaw_rad, friction, damping)
    print("pre_v_2d:", pre_v_2d)
    print("post_v_2d:", post_v_2d)

if __name__ == "__main__":
    _demo()