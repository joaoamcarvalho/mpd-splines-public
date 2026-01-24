import math
import os
from pathlib import Path
from typing import Sequence, Optional

import numpy as np
import torch
import torch.nn as nn

try:
    from scipy import interpolate
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

import matplotlib.pyplot as plt


def _bspline_open_uniform_knots_torch(n_ctrl: int, degree: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create clamped open-uniform knot vector matching recons_average.py.

    knots = [0]*degree + linspace(0,1,n_ctrl-degree+1) + [1]*degree
    length = n_ctrl + degree + 1
    """
    if n_ctrl <= degree:
        raise ValueError(f"n_ctrl ({n_ctrl}) must be > degree ({degree})")
    step = torch.linspace(0.0, 1.0, n_ctrl - degree + 1, device=device, dtype=dtype)
    head = torch.zeros(degree, device=device, dtype=dtype)
    tail = torch.ones(degree, device=device, dtype=dtype)
    return torch.cat([head, step, tail], dim=0)


def _bspline_deboor_eval_torch(ctrl_pts: torch.Tensor, u: torch.Tensor, degree: int, knots: torch.Tensor) -> torch.Tensor:
    """Evaluate clamped B-spline at parameter u using de Boor (batched).

    ctrl_pts: (B, N, 2)
    u: (B,) or scalar tensor
    knots: (N + degree + 1,)
    returns: (B, 2)
    """
    if ctrl_pts.dim() != 3 or ctrl_pts.size(-1) != 2:
        raise ValueError(f"Expected ctrl_pts (B,N,2), got {tuple(ctrl_pts.shape)}")
    B, N, _ = ctrl_pts.shape
    p = int(degree)
    if N <= p:
        raise ValueError("Control point count must be > degree")

    if u.dim() == 0:
        u = u.expand(B)
    elif u.dim() == 1 and u.numel() == 1:
        u = u.expand(B)
    elif u.dim() != 1 or u.numel() != B:
        raise ValueError(f"Expected u as scalar or (B,), got {tuple(u.shape)}")

    # Clamp u into [0,1] with a tiny epsilon to avoid falling off the end.
    u = u.clamp(0.0, 1.0)
    eps = torch.as_tensor(1e-7, device=ctrl_pts.device, dtype=ctrl_pts.dtype)
    u_safe = torch.where(u >= 1.0, (1.0 - eps).expand_as(u), u)

    # Find knot span k: knots[k] <= u < knots[k+1]
    # For u==1, k should be N-1.
    k = torch.searchsorted(knots, u_safe, right=False) - 1
    k = k.clamp(min=p, max=N - 1)

    # Initialize de Boor points d[j] = P[k-p+j], j=0..p
    j0 = torch.arange(0, p + 1, device=ctrl_pts.device)
    idx = (k[:, None] - p + j0[None, :]).clamp(0, N - 1)  # (B, p+1)
    batch_idx = torch.arange(B, device=ctrl_pts.device)[:, None]
    d = ctrl_pts[batch_idx, idx].clone()  # (B, p+1, 2)

    # de Boor recursion
    for r in range(1, p + 1):
        for j in range(p, r - 1, -1):
            i = k - p + j
            t_i = knots[i]
            t_ip = knots[i + p - r + 1]
            denom = (t_ip - t_i)
            denom = torch.where(denom.abs() < eps, eps.expand_as(denom), denom)
            alpha = (u_safe - t_i) / denom
            alpha = alpha.unsqueeze(-1)  # (B,1)
            d[:, j, :] = (1.0 - alpha) * d[:, j - 1, :] + alpha * d[:, j, :]

    return d[:, p, :]


def _estimate_v_avg_endpoint_torch(ctrl_pre: torch.Tensor, T_pre: torch.Tensor, degree: int, spans: int = 2) -> torch.Tensor:
    """Local-average endpoint velocity v_avg as in recons_average.py (no SciPy).

    v_avg = (C(1) - C(1-delta_u)) / (T_pre * delta_u)
    delta_u = spans * h, h = 1/(N-degree)

    ctrl_pre: (B, N, 2)
    T_pre: (B,) or (B,1)
    returns: (B,2)
    """
    if T_pre.dim() == 2 and T_pre.size(1) == 1:
        T_pre = T_pre.squeeze(1)
    if T_pre.dim() != 1:
        raise ValueError(f"Expected T_pre (B,) or (B,1), got {tuple(T_pre.shape)}")

    B, N, _ = ctrl_pre.shape
    p = int(degree)
    if N <= p:
        return torch.zeros((B, 2), device=ctrl_pre.device, dtype=ctrl_pre.dtype)

    h = 1.0 / float(N - p)
    delta_u = float(spans) * h
    if delta_u >= 1.0:
        delta_u = 1.0 - 1e-6

    pos1 = ctrl_pre[:, -1, :]
    u0 = torch.as_tensor(1.0 - delta_u, device=ctrl_pre.device, dtype=ctrl_pre.dtype)
    knots = _bspline_open_uniform_knots_torch(N, p, device=ctrl_pre.device, dtype=ctrl_pre.dtype)
    pos0 = _bspline_deboor_eval_torch(ctrl_pre, u0, p, knots)

    denom = (T_pre * delta_u).clamp(min=1e-6)
    return (pos1 - pos0) / denom.unsqueeze(-1)

# ----------------- 模型超参 & 归一化常数（与当前训练一致） -----------------

FULL_HORIZON = 21
MODEL_HORIZON = 12
STATE_DIM = 2

# 这些超参在 checkpoint 中无法完整推断（例如 num_heads/cross_attn_layers 不在 state_dict 里），
# 因此保持与当前训练脚本一致。
HIDDEN_DIM = 128
NUM_LAYERS = 4
NUM_HEADS = 8
DROPOUT = 0.1
N_CTX_TOKENS = 4
CROSS_ATTN_LAYERS = 2

# B 样条参数（与训练/数据生成保持一致）
BSPLINE_DEGREE = 5
T_DURATION = 4.0

# 这些是你在 train_student_distill.py 里用的 std
POS_STD = 2.4662630558013916 #1.820559 
VEL_STD = 1.5834906101226807 #0.842956
ACC_STD = 2.285597085952759 #1.718903

# ----------------- StudentTransformer 定义（从 small/student_model.py 拷贝） -----------------

def _init_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        if getattr(module, "weight", None) is not None:
            nn.init.xavier_uniform_(module.weight)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        if getattr(module, "weight", None) is not None:
            nn.init.ones_(module.weight)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)


def sinusoidal_position_embedding(length: int, dim: int, device=None) -> torch.Tensor:
    position = torch.arange(length, device=device).unsqueeze(1)  # (L,1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * -(math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1, use_cross: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_cross = use_cross
        self.num_heads = num_heads

        self.ln1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.ln2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6) if use_cross else None
        self.ln3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)

        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout)

        if use_cross:
            self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout)

        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

        self.ada_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim)
        )
        if isinstance(self.ada_proj[-1], nn.Linear):
            nn.init.zeros_(self.ada_proj[-1].weight)
            if self.ada_proj[-1].bias is not None:
                nn.init.zeros_(self.ada_proj[-1].bias)

    def forward(self, x: torch.Tensor, context_vec: torch.Tensor, context_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, D = x.shape

        params = self.ada_proj(context_vec)  # [B, 6*D]
        p = params.view(B, 3, 2, D)
        shift1, scale1 = p[:, 0, 0], p[:, 0, 1]
        shift2, scale2 = p[:, 1, 0], p[:, 1, 1]
        shift3, scale3 = p[:, 2, 0], p[:, 2, 1]

        x_norm = self.ln1(x)
        x_norm = modulate(x_norm, shift1, scale1)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out

        if self.use_cross and context_tokens is not None:
            x_norm2 = self.ln2(x)
            x_norm2 = modulate(x_norm2, shift2, scale2)
            cross_out, _ = self.cross_attn(x_norm2, context_tokens, context_tokens, need_weights=False)
            x = x + cross_out

        x_norm3 = self.ln3(x)
        x_norm3 = modulate(x_norm3, shift3, scale3)
        mlp_out = self.mlp(x_norm3)
        x = x + mlp_out

        return x


class StudentTransformer(nn.Module):
    def __init__(
        self,
        n_control_points: int = MODEL_HORIZON,
        state_dim: int = STATE_DIM,
        context_dim: int = 128,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        num_heads: int = NUM_HEADS,
        dropout: float = DROPOUT,
        pos_scale: float = 0.1,  # 保留参数以兼容，但在 learnable queries 下不使用
        n_ctx_tokens: int = N_CTX_TOKENS,
        cross_attn_layers: int = CROSS_ATTN_LAYERS,
    ):
        super().__init__()
        self.n_control_points = n_control_points
        self.state_dim = state_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.n_ctx_tokens = n_ctx_tokens
        self.cross_attn_layers = max(0, min(num_layers, cross_attn_layers))

        # 1. 增强 Context 投影 (MLP)
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.context_to_tokens = nn.Linear(context_dim, n_ctx_tokens * hidden_dim)

        # 2. Learnable Queries (核心：给每个点独立的身份)
        # 形状: [1, N, hidden]
        self.query_embed = nn.Parameter(torch.randn(1, n_control_points, hidden_dim))

        # Transformer Blocks
        blocks = []
        for i in range(num_layers):
            layer_idx_from_end = num_layers - 1 - i
            use_cross = layer_idx_from_end < self.cross_attn_layers
            blocks.append(CrossAttentionBlock(hidden_dim, num_heads, mlp_ratio=4.0, dropout=dropout, use_cross=use_cross))
        self.blocks = nn.ModuleList(blocks)

        # ============================================================
        # 3. 解耦输出头 (Decoupled Heads)
        # ============================================================
        
        # 设定分割点：最后 2 个点专用一个头，其余点用另一个头
        self.n_keypoints = 2 
        self.n_body = n_control_points - self.n_keypoints

        # Head A: Body (负责整体形状，平滑性)
        self.head_body = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, state_dim)
        )

        # Head B: Keypoints (负责最后两个高权重关键点)
        self.head_keypoint = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, state_dim)
        )

        # 初始化输出层 (使其初始输出接近0)
        for m in [self.head_body, self.head_keypoint]:
            nn.init.uniform_(m[-1].weight, -0.001, 0.001)
            nn.init.zeros_(m[-1].bias)

        self.apply(_init_weights)

    def forward(self, context: torch.Tensor, return_hidden: bool = False):
        B = context.size(0)
        
        # 1. Context Processing
        ctx_global = self.context_proj(context)
        ctx_tok = self.context_to_tokens(context).view(B, self.n_ctx_tokens, self.hidden_dim)

        # 2. Expand Queries
        x = self.query_embed.expand(B, -1, -1)

        # 3. Backbone
        hidden_states = []
        for blk in self.blocks:
            x = blk(x, ctx_global, ctx_tok if blk.use_cross else None)
            hidden_states.append(x)
        
        # 4. Decoupled Output Projection
        x_body = x[:, :self.n_body, :]
        x_key  = x[:, self.n_body:, :]

        out_body = self.head_body(x_body)
        out_key  = self.head_keypoint(x_key)

        out = torch.cat([out_body, out_key], dim=1)

        if return_hidden:
            return out, hidden_states
        return out

# ----------------- 归一化函数 -----------------
def normalize_position_group_torch(x: torch.Tensor) -> torch.Tensor:
    std = torch.as_tensor(POS_STD, dtype=x.dtype, device=x.device)
    return x / std


def unnormalize_position_group_torch(x: torch.Tensor) -> torch.Tensor:
    std = torch.as_tensor(POS_STD, dtype=x.dtype, device=x.device)
    return x * std


def normalize_velocity_group_torch(x: torch.Tensor) -> torch.Tensor:
    std = torch.as_tensor(VEL_STD, dtype=x.dtype, device=x.device)
    return x / std


def unnormalize_velocity_group_torch(x: torch.Tensor) -> torch.Tensor:
    std = torch.as_tensor(VEL_STD, dtype=x.dtype, device=x.device)
    return x * std


def normalize_acceleration_group_torch(x: torch.Tensor) -> torch.Tensor:
    std = torch.as_tensor(ACC_STD, dtype=x.dtype, device=x.device)
    return x / std


def unnormalize_acceleration_group_torch(x: torch.Tensor) -> torch.Tensor:
    std = torch.as_tensor(ACC_STD, dtype=x.dtype, device=x.device)
    return x * std


def normalize_control_points_torch(x: torch.Tensor) -> torch.Tensor:
    return normalize_position_group_torch(x)


def unnormalize_control_points_torch(x: torch.Tensor) -> torch.Tensor:
    return unnormalize_position_group_torch(x)


def reconstruct_start_control_points(sp, sv, sa, T, degree, n_control_points):
    # sp, sv, sa: (B, 2)
    # T: float or (B,) or (B,1)
    device = sp.device
    dtype = sp.dtype
    k = float(degree)

    if isinstance(T, float):
        T_tensor = torch.tensor(T, dtype=dtype, device=device)
    else:
        T_tensor = T

    # Make endpoint reconstruction consistent with clamped open-uniform knots.
    # We solve using derivative-control-point relations under u=t/T:
    #   Q0 = dC/du(0+) = T * v0
    #   R0 = d2C/du2(0+) = T^2 * a0
    # with clamped knots U.
    # For the first two derivative control points:
    #   Q0 = p*(P1-P0)/(U[p+1]-U[1])
    #   R0 = (p-1)*(Q1-Q0)/(U[p+1]-U[2])
    #   Q1 = p*(P2-P1)/(U[p+2]-U[2])
    knots = _bspline_open_uniform_knots_torch(n_control_points, degree, device=device, dtype=dtype)

    def _as_scale(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 0:
            return x
        if x.dim() == 1:
            return x.unsqueeze(-1)
        if x.dim() == 2 and x.size(1) == 1:
            return x
        return x

    T_scale = _as_scale(T_tensor)
    Q0 = sv * T_scale
    R0 = sa * (T_scale ** 2)

    den_Q0 = (knots[degree + 1] - knots[1]).clamp(min=1e-12)
    den_R0 = (knots[degree + 1] - knots[2]).clamp(min=1e-12)
    den_Q1 = (knots[degree + 2] - knots[2]).clamp(min=1e-12)

    C0 = sp
    C1 = C0 + Q0 * (den_Q0 / k)
    Q1 = Q0 + R0 * (den_R0 / (k - 1.0))
    C2 = C1 + Q1 * (den_Q1 / k)

    return torch.stack([C0, C1, C2], dim=1)


def reconstruct_end_control_points(ep, ev, ea, T, degree, n_control_points):
    # ep, ev, ea: (B, 2)
    device = ep.device
    dtype = ep.dtype
    k = float(degree)

    if isinstance(T, float):
        T_tensor = torch.tensor(T, dtype=dtype, device=device)
    else:
        T_tensor = T

    # Symmetric endpoint reconstruction at u->1- consistent with clamped open-uniform knots.
    knots = _bspline_open_uniform_knots_torch(n_control_points, degree, device=device, dtype=dtype)

    def _as_scale(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 0:
            return x
        if x.dim() == 1:
            return x.unsqueeze(-1)
        if x.dim() == 2 and x.size(1) == 1:
            return x
        return x

    T_scale = _as_scale(T_tensor)
    Q_last = ev * T_scale
    R_last = ea * (T_scale ** 2)

    den_Q_last = (knots[n_control_points + degree - 1] - knots[n_control_points - 1]).clamp(min=1e-12)
    den_R_last = (knots[n_control_points + degree - 2] - knots[n_control_points - 1]).clamp(min=1e-12)
    den_Q_prev = (knots[n_control_points + degree - 2] - knots[n_control_points - 2]).clamp(min=1e-12)

    Cn_1 = ep
    Cn_2 = Cn_1 - Q_last * (den_Q_last / k)
    Q_prev = Q_last - R_last * (den_R_last / (k - 1.0))
    Cn_3 = Cn_2 - Q_prev * (den_Q_prev / k)

    return torch.stack([Cn_3, Cn_2, Cn_1], dim=1)


def remap_model_outputs_standalone(x_norm: torch.Tensor, orig_horizon: int, context_d: dict, degree: int, T_duration: float) -> torch.Tensor:
    """Standalone equivalent of mpd.models.diffusion_models.sample_functions.remap_model_outputs.

    x_norm: (B, out_N, 2) normalized control points.
    Returns: (B, orig_horizon, 2) physical control points (unnormalized).
    """
    B, out_N, D = x_norm.shape
    device = x_norm.device

    tm = context_d["task_mask"].to(device)
    sp_phys = context_d["start_pos_phys"].to(device)
    sv_phys = context_d["start_vel_phys"].to(device)
    sa_phys = context_d["start_acc_phys"].to(device)
    geom_phys = context_d["geometry_offset_phys"].to(device)
    
    x_phys = unnormalize_control_points_torch(x_norm)

    # Check for uniform batch
    is_avoidance = (tm[:, 0] < 0.5)
    
    if is_avoidance.all():
        # Avoidance: Return 18 points
        ep_phys = sp_phys + geom_phys
        ev_phys = torch.zeros_like(sv_phys)
        ea_phys = torch.zeros_like(sa_phys)

        C_start = reconstruct_start_control_points(sp_phys, sv_phys, sa_phys, T_duration, degree, orig_horizon - 3)
        C_end = reconstruct_end_control_points(ep_phys, ev_phys, ea_phys, T_duration, degree, orig_horizon - 3)

        # x_phys is (B, 12, 2).
        res = torch.cat([C_start, x_phys, C_end], dim=1)
        lengths = torch.full((B,), 18, dtype=torch.long, device=device)
        return res, lengths

    elif (~is_avoidance).all():
        # Collision: Return 21 points
        guide_vec_phys = context_d["guide_vector_phys"].to(device)
        collision_point = sp_phys + geom_phys

        last_pred = x_phys[:, -1, :]
        L_post = torch.norm(last_pred - collision_point, dim=-1)
        v_post_mag = torch.norm(guide_vec_phys, dim=-1)
        eps = 1e-6
        t_post_vals = 2.0 * L_post / (v_post_mag + eps)

        ctrl_collision = collision_point
        n_pre = max(orig_horizon - 5, degree + 1)
        k = float(degree)
        # T_pre = T_duration - t_post_vals (per-sample)
        T_pre = (torch.as_tensor(T_duration, dtype=t_post_vals.dtype, device=device) - t_post_vals).clamp(min=1e-6)

        # v_der0: initialize from the model-predicted last interior control point (keeps legacy behavior)
        ctrl_pre_prev_init = x_phys[:, -2, :]
        d_pre = (ctrl_collision - ctrl_pre_prev_init)  # (B,2)
        delta_u_pre = 1.0 / float(n_pre - degree)
        factor_v_pre = (T_pre * delta_u_pre) / k
        v_der0 = d_pre / (factor_v_pre.unsqueeze(-1) + 1e-6)

        # v_avg: local-average endpoint velocity over last `spans` knot spans (no SciPy)
        spans = 2
        w = 0.001
        end_acc_zeros = torch.zeros_like(sa_phys)

        # Precompute prefix control points once
        C_start_collision = reconstruct_start_control_points(sp_phys, sv_phys, sa_phys, T_pre, degree, orig_horizon - 5)

        # Fixed-point iteration for self-consistent v_pre_vec since ctrl_pre16 tail depends on v_pre_vec.
        v_pre_vec = v_der0
        for _ in range(3):
            tail_pre = reconstruct_end_control_points(ctrl_collision, v_pre_vec, end_acc_zeros, T_pre, degree, n_pre)  # (B,3,2)
            ctrl_pre16 = torch.cat([C_start_collision, x_phys[:, :10, :], tail_pre], dim=1)
            v_avg = _estimate_v_avg_endpoint_torch(ctrl_pre16, T_pre, degree=degree, spans=spans)
            v_pre_vec = w * v_der0 + (1.0 - w) * v_avg
        print(f"v_pre_vel: {v_pre_vec}")
        tail_pre = reconstruct_end_control_points(ctrl_collision, v_pre_vec, end_acc_zeros, T_pre, degree, n_pre)
        ctrl_pre_pre_prev = tail_pre[:, 0, :]
        ctrl_pre_prev = tail_pre[:, 1, :]

        end_vel_zeros = torch.zeros_like(sv_phys)
        end_acc_zeros = torch.zeros_like(sa_phys)
        start_acc_zeros = torch.zeros_like(sa_phys)

        Cpost_start = reconstruct_start_control_points(collision_point, guide_vec_phys, start_acc_zeros, t_post_vals, degree, 6)
        Cpost_end = reconstruct_end_control_points(last_pred, end_vel_zeros, end_acc_zeros, t_post_vals, degree, 6)
        Cpost = torch.cat([Cpost_start, Cpost_end], dim=1)

        res = torch.cat([
            C_start_collision,
            x_phys[:, :10, :],
            ctrl_pre_pre_prev.unsqueeze(1),
            ctrl_pre_prev.unsqueeze(1),
            Cpost
        ], dim=1)
        lengths = torch.full((B,), 21, dtype=torch.long, device=device)
        return res, lengths

    else:
        # Mixed batch
        new_x_phys = torch.zeros((B, orig_horizon, D), device=device, dtype=x_phys.dtype)
        lengths = torch.full((B,), int(orig_horizon), dtype=torch.long, device=device)
        
        avoid_mask = is_avoidance
        collision_mask = ~is_avoidance
        
        if avoid_mask.any():
            sp_sel = sp_phys[avoid_mask]
            sv_sel = sv_phys[avoid_mask]
            sa_sel = sa_phys[avoid_mask]
            geom_sel = geom_phys[avoid_mask]
            
            ep_sel = sp_sel + geom_sel
            ev_sel = torch.zeros_like(sv_sel)
            ea_sel = torch.zeros_like(sa_sel)
            
            C_start = reconstruct_start_control_points(sp_sel, sv_sel, sa_sel, T_duration, degree, orig_horizon - 3)
            C_end = reconstruct_end_control_points(ep_sel, ev_sel, ea_sel, T_duration, degree, orig_horizon - 3)
            
            new_x_phys[avoid_mask, 0:3, :] = C_start
            new_x_phys[avoid_mask, 3:15, :] = x_phys[avoid_mask]
            new_x_phys[avoid_mask, 15:18, :] = C_end
            lengths[avoid_mask] = 18
            
        if collision_mask.any():
            sp_c = sp_phys[collision_mask]
            sv_c = sv_phys[collision_mask]
            sa_c = sa_phys[collision_mask]
            geom_c = geom_phys[collision_mask]
            guide_sel = context_d["guide_vector_phys"].to(device)[collision_mask]
            
            collision_point = sp_c + geom_c
            last_pred = x_phys[collision_mask, -1, :]
            
            L_post = torch.norm(last_pred - collision_point, dim=-1)
            v_post_mag = torch.norm(guide_sel, dim=-1)
            eps = 1e-6
            t_post_vals = 2.0 * L_post / (v_post_mag + eps)
            
            ctrl_collision = collision_point
            n_pre = max(orig_horizon - 5, degree + 1)
            k = float(degree)
            T_pre = (torch.as_tensor(T_duration, dtype=t_post_vals.dtype, device=device) - t_post_vals).clamp(min=1e-6)

            # v_der0: initialize from model-predicted last interior control point (sub-batch)
            ctrl_pre_prev_init = x_phys[collision_mask, -2, :]
            d_pre = (ctrl_collision - ctrl_pre_prev_init)
            delta_u_pre = 1.0 / float(n_pre - degree)
            factor_v_pre = (T_pre * delta_u_pre) / k
            v_der0 = d_pre / (factor_v_pre.unsqueeze(-1) + 1e-6)

            spans = 2
            w = 0.001
            end_acc_zeros = torch.zeros_like(sa_c)

            # Prefix control points for this sub-batch
            C_start_collision = reconstruct_start_control_points(sp_c, sv_c, sa_c, T_pre, degree, orig_horizon - 5)

            v_pre_vec = v_der0
            for _ in range(3):
                tail_pre = reconstruct_end_control_points(ctrl_collision, v_pre_vec, end_acc_zeros, T_pre, degree, n_pre)  # (Bc,3,2)
                ctrl_pre16 = torch.cat([C_start_collision, x_phys[collision_mask, :10, :], tail_pre], dim=1)
                v_avg = _estimate_v_avg_endpoint_torch(ctrl_pre16, T_pre, degree=degree, spans=spans)
                v_pre_vec = w * v_der0 + (1.0 - w) * v_avg

            tail_pre = reconstruct_end_control_points(ctrl_collision, v_pre_vec, end_acc_zeros, T_pre, degree, n_pre)
            ctrl_pre_pre_prev = tail_pre[:, 0, :]
            ctrl_pre_prev = tail_pre[:, 1, :]
            
            end_vel_zeros = torch.zeros_like(sv_c)
            end_acc_zeros = torch.zeros_like(sa_c)
            start_acc_zeros = torch.zeros_like(sa_c)
            
            Cpost_start = reconstruct_start_control_points(collision_point, guide_sel, start_acc_zeros, t_post_vals, degree, 6)
            Cpost_end = reconstruct_end_control_points(last_pred, end_vel_zeros, end_acc_zeros, t_post_vals, degree, 6)
            Cpost = torch.cat([Cpost_start, Cpost_end], dim=1)
            
            new_x_phys[collision_mask, 0:3, :] = C_start_collision
            new_x_phys[collision_mask, 3:13, :] = x_phys[collision_mask, :10, :]
            new_x_phys[collision_mask, 13:14, :] = ctrl_pre_pre_prev.unsqueeze(1)
            new_x_phys[collision_mask, 14:15, :] = ctrl_pre_prev.unsqueeze(1)
            new_x_phys[collision_mask, orig_horizon - 6 : orig_horizon, :] = Cpost
            
        return new_x_phys, lengths

def bspline_trajectory(control_points: np.ndarray, num_points: int = 128, degree: int = 5) -> np.ndarray:
    """
    给定控制点 (N,2)，返回 B 样条轨迹采样点 (num_points,2)。
    若环境无 SciPy，则直接线性插值连线。
    """
    control_points = np.asarray(control_points, dtype=np.float32)
    N, D = control_points.shape
    assert D == 2

    if not HAS_SCIPY or N <= degree:
        # fallback: 线性插值
        t = np.linspace(0.0, 1.0, num_points)
        idx_float = t * (N - 1)
        idx0 = np.floor(idx_float).astype(int)
        idx1 = np.minimum(idx0 + 1, N - 1)
        w = (idx_float - idx0)[:, None]
        return (1 - w) * control_points[idx0] + w * control_points[idx1]

    # 与仓库中生成脚本相同的 knots 形式
    c = N
    d = degree
    knots = np.pad(np.linspace(0.0, 1.0, c + 1 - d), (d, d), "constant", constant_values=(0.0, 1.0))
    cc = control_points.T  # (2,N)
    u = np.linspace(0.0, 1.0, num_points)
    traj = interpolate.BSpline(knots, cc.T, d)(u)  # (num_points,2)
    return np.asarray(traj, dtype=np.float32)


def _bspline_open_uniform_knots_np(n_ctrl: int, degree: int) -> np.ndarray:
    """Open-uniform clamped knot vector, consistent with SciPy usage in this file."""
    if n_ctrl <= degree:
        raise ValueError(f"n_ctrl ({n_ctrl}) must be > degree ({degree})")
    knots = np.pad(
        np.linspace(0.0, 1.0, n_ctrl + 1 - degree, dtype=np.float64),
        (degree, degree),
        mode="constant",
        constant_values=(0.0, 1.0),
    )
    return knots


def _bspline_find_span(n_ctrl: int, degree: int, u: float, knots: np.ndarray) -> int:
    """Find knot span index k such that knots[k] <= u < knots[k+1]."""
    p = int(degree)
    n = int(n_ctrl)
    if u >= knots[n]:
        return n - 1
    if u <= knots[p]:
        return p
    k = int(np.searchsorted(knots, u, side="right") - 1)
    return int(np.clip(k, p, n - 1))


def _bspline_deboor_eval_np(ctrl_pts: np.ndarray, u: float, degree: int, knots: np.ndarray) -> np.ndarray:
    """Evaluate clamped B-spline curve point using de Boor (numpy, single u)."""
    P = np.asarray(ctrl_pts, dtype=np.float64)
    n, d = P.shape
    p = int(degree)
    if n <= p:
        raise ValueError("Control point count must be > degree")

    u_clamped = float(np.clip(u, 0.0, 1.0))
    # Avoid falling off the end in span selection
    eps = 1e-12
    u_safe = (1.0 - eps) if u_clamped >= 1.0 else u_clamped

    k = _bspline_find_span(n, p, u_safe, knots)
    dpts = P[(k - p) : (k + 1), :].copy()  # (p+1, d)

    for r in range(1, p + 1):
        for j in range(p, r - 1, -1):
            i = k - p + j
            denom = knots[i + p - r + 1] - knots[i]
            if abs(denom) < eps:
                alpha = 0.0
            else:
                alpha = (u_safe - knots[i]) / denom
            dpts[j, :] = (1.0 - alpha) * dpts[j - 1, :] + alpha * dpts[j, :]

    return dpts[p, :]


def _bspline_derivative_ctrl_pts_np(ctrl_pts: np.ndarray, degree: int, knots: np.ndarray, order: int = 1) -> tuple[np.ndarray, int, np.ndarray]:
    """Compute derivative B-spline control points for d^order C / du^order.

    Returns (ctrl_der, degree_der, knots_der).
    """
    if order < 0:
        raise ValueError("order must be >= 0")
    P = np.asarray(ctrl_pts, dtype=np.float64)
    p = int(degree)
    U = np.asarray(knots, dtype=np.float64)

    if order == 0:
        return P, p, U
    if order > p:
        # higher derivatives are zero
        return np.zeros((max(P.shape[0] - order, 1), P.shape[1]), dtype=np.float64), 0, U

    Pk = P
    pk = p
    Uk = U
    for _ in range(order):
        n_ctrl = Pk.shape[0]
        if n_ctrl <= 1 or pk <= 0:
            return np.zeros((1, P.shape[1]), dtype=np.float64), 0, Uk

        denom = (Uk[(1 + pk) : (n_ctrl + pk)] - Uk[1:n_ctrl])  # length n_ctrl-1
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        Pk = (pk * (Pk[1:, :] - Pk[:-1, :])) / denom[:, None]
        Uk = Uk[1:-1]
        pk = pk - 1

    return Pk, pk, Uk


def bspline_derivatives_np(control_points: np.ndarray, T: float, num_points: int = 128, degree: int = 5):
    """
    返回位置、速度、加速度及对应时间数组：pos (num_points,2), vel (num_points,2), acc (num_points,2), t (num_points,)
    如果有 SciPy，则使用解析导数；否则用数值差分。
    """
    cps = np.asarray(control_points, dtype=np.float64)
    N, D = cps.shape
    t = np.linspace(0.0, T, num_points)

    # Evaluate basis and its derivatives at parameter u = t/T (clamped open-uniform knots)
    p = int(degree)
    if N <= p:
        # degenerate: fallback to linear samples
        pos = bspline_trajectory(cps, num_points=num_points, degree=degree).astype(np.float64)
        vel = np.zeros_like(pos)
        acc = np.zeros_like(pos)
        return pos.astype(np.float32), vel.astype(np.float32), acc.astype(np.float32), t.astype(np.float32)

    # protect T
    T_eff = float(T) if float(T) > 1e-9 else 1e-9

    knots = _bspline_open_uniform_knots_np(N, p)
    d1_ctrl, d1_deg, d1_knots = _bspline_derivative_ctrl_pts_np(cps, p, knots, order=1)
    d2_ctrl, d2_deg, d2_knots = _bspline_derivative_ctrl_pts_np(cps, p, knots, order=2)

    # NOTE: for derivatives at endpoints, use one-sided limits (u->0+, u->1-)
    u_vals = np.linspace(0.0, 1.0, num_points)
    eps_u = 1e-9
    pos = np.zeros((num_points, D), dtype=np.float64)
    vel = np.zeros((num_points, D), dtype=np.float64)
    acc = np.zeros((num_points, D), dtype=np.float64)

    for idx, u in enumerate(u_vals):
        u_pos = float(u)
        if u_pos <= 0.0:
            pos_u = cps[0]
        elif u_pos >= 1.0:
            pos_u = cps[-1]
        else:
            pos_u = _bspline_deboor_eval_np(cps, u_pos, p, knots)

        u_der = float(min(max(u_pos, eps_u), 1.0 - eps_u))
        C_du = _bspline_deboor_eval_np(d1_ctrl, u_der, d1_deg, d1_knots)
        C_d2u = _bspline_deboor_eval_np(d2_ctrl, u_der, d2_deg, d2_knots)

        pos[idx, :] = pos_u
        vel[idx, :] = C_du / T_eff
        acc[idx, :] = C_d2u / (T_eff ** 2)

    t = u_vals * T_eff
    return pos.astype(np.float32), vel.astype(np.float32), acc.astype(np.float32), t.astype(np.float32)


def _bspline_basis_and_derivatives_np(n_ctrl: int, degree: int, u: float) -> tuple:
    """Compute B-spline basis and first/second derivatives (w.r.t. parameter u) at scalar u.

    Returns (basis, dbasis, d2basis) each arrays of shape (n_ctrl,).
    Uses open-uniform/clamped knots consistent with other functions.
    """
    p = int(degree)
    n = int(n_ctrl)
    # knots: [0]*p + linspace(0,1,n-p+1) + [1]*p
    knots = np.concatenate([np.zeros(p), np.linspace(0.0, 1.0, n - p + 1), np.ones(p)])

    # allocate
    basis = np.zeros((p + 1, n), dtype=np.float64)
    dbasis = np.zeros((p + 1, n), dtype=np.float64)
    d2basis = np.zeros((p + 1, n), dtype=np.float64)

    # degree 0
    for i in range(n):
        t0 = knots[i]
        t1 = knots[i + 1]
        if (u >= t0 and u < t1) or (u == 1.0 and t1 == 1.0 and i == n - 1):
            basis[0, i] = 1.0

    # build up
    for d in range(1, p + 1):
        for i in range(n):
            left_denom = knots[i + d] - knots[i]
            right_denom = knots[i + d + 1] - knots[i + 1]

            left = 0.0
            if left_denom > 0:
                left = (u - knots[i]) / left_denom * basis[d - 1, i]
            right = 0.0
            if right_denom > 0 and i + 1 < n:
                right = (knots[i + d + 1] - u) / right_denom * basis[d - 1, i + 1]
            basis[d, i] = left + right

            # first derivative using Cox-de Boor derivative formula
            a = 0.0
            if left_denom > 0:
                a = basis[d - 1, i] / left_denom
            b = 0.0
            if right_denom > 0 and i + 1 < n:
                b = basis[d - 1, i + 1] / right_denom
            dbasis[d, i] = d * (a - b)

            # second derivative: derivative of dbasis uses dbasis of degree d-1
            a2 = 0.0
            if left_denom > 0:
                a2 = dbasis[d - 1, i] / left_denom
            b2 = 0.0
            if right_denom > 0 and i + 1 < n:
                b2 = dbasis[d - 1, i + 1] / right_denom
            d2basis[d, i] = d * (a2 - b2)

    return basis[p, :], dbasis[p, :], d2basis[p, :]


def plot_vel_acc_for_control_points(control_points: np.ndarray, T_pre: float, num_points: int = 256, degree: int = 5, save_path: Optional[Path] = None, show: bool = False):
    """
    为给定的控制点（例如 16 个碰撞前控制点）绘制速度与加速度（分量与模）。
    """
    pos, vel, acc, t = bspline_derivatives_np(control_points, T_pre, num_points=num_points, degree=degree)

    speed = np.linalg.norm(vel, axis=1)
    acc_mag = np.linalg.norm(acc, axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(t, vel[:, 0], label="vx")
    axes[0].plot(t, vel[:, 1], label="vy")
    axes[0].plot(t, speed, label="|v|", linestyle="--", color="k")
    axes[0].set_ylabel("Velocity (m/s)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True)

    axes[1].plot(t, acc[:, 0], label="ax")
    axes[1].plot(t, acc[:, 1], label="ay")
    axes[1].plot(t, acc_mag, label="|a|", linestyle="--", color="k")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Acceleration (m/s^2)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True)

    fig.suptitle("Velocity and Acceleration (pre-collision B-spline)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def compute_fd_vel_acc(traj: np.ndarray, T: float):
    """
    使用有限差分（中心差分内点，端点用一侧差分）计算给定采样轨迹的速度和加速度。

    traj: (M,2) 采样点
    T: 总时长
    返回 (vel (M,2), acc (M,2), t (M,))
    """
    pts = np.asarray(traj, dtype=np.float64)
    M = pts.shape[0]
    if M < 2:
        return np.zeros_like(pts), np.zeros_like(pts), np.linspace(0.0, T, M)
    dt = float(T) / float(max(1, M - 1))
    vel = np.zeros_like(pts)
    acc = np.zeros_like(pts)

    # velocity: central differences for interior, forward/backward for boundaries
    if M >= 2:
        vel[0] = (pts[1] - pts[0]) / dt
        vel[-1] = (pts[-1] - pts[-2]) / dt
    for i in range(1, M - 1):
        vel[i] = (pts[i + 1] - pts[i - 1]) / (2.0 * dt)

    # acceleration: central second difference for interior, one-sided for boundaries
    if M >= 3:
        acc[0] = (pts[2] - 2.0 * pts[1] + pts[0]) / (dt * dt)
        acc[-1] = (pts[-1] - 2.0 * pts[-2] + pts[-3]) / (dt * dt)
    elif M == 2:
        acc[0] = acc[-1] = (pts[1] - 2.0 * pts[0] + pts[0]) / (dt * dt)
    for i in range(1, M - 1):
        acc[i] = (pts[i + 1] - 2.0 * pts[i] + pts[i - 1]) / (dt * dt)

    t = np.linspace(0.0, T, M)
    return vel.astype(np.float32), acc.astype(np.float32), t.astype(np.float32)


def plot_fd_vel_acc_for_control_points(control_points: np.ndarray, T_pre: float, num_points: int = 256, degree: int = 5, save_path: Optional[Path] = None, show: bool = False):
    """
    用有限差分在采样轨迹上绘制速度/加速度（分量与模）。
    """
    # 先用 bspline_trajectory 取得采样点
    traj = bspline_trajectory(control_points, num_points=num_points, degree=degree)
    vel, acc, t = compute_fd_vel_acc(traj, T_pre)

    speed = np.linalg.norm(vel, axis=1)
    acc_mag = np.linalg.norm(acc, axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(t, vel[:, 0], label="vx_fd")
    axes[0].plot(t, vel[:, 1], label="vy_fd")
    axes[0].plot(t, speed, label="|v|_fd", linestyle="--", color="k")
    axes[0].set_ylabel("Velocity (m/s)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True)

    axes[1].plot(t, acc[:, 0], label="ax_fd")
    axes[1].plot(t, acc[:, 1], label="ay_fd")
    axes[1].plot(t, acc_mag, label="|a|_fd", linestyle="--", color="k")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Acceleration (m/s^2)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True)

    fig.suptitle("Finite-Difference Velocity and Acceleration (pre-collision B-spline)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _print_endpoint_states(tag: str, control_points: np.ndarray, T_pre: float, degree: int, start_pos: np.ndarray, start_vel: np.ndarray, start_acc: np.ndarray, end_pos: np.ndarray, end_vel: np.ndarray, end_acc: np.ndarray):
    """Debug helper: compare evaluated endpoint states vs desired."""
    _, vel0, acc0, _ = bspline_derivatives_np(control_points, T_pre, num_points=3, degree=degree)
    # num_points=3 uses u=[0,0.5,1], endpoints are index 0 and -1
    pos = bspline_trajectory(control_points, num_points=3, degree=degree)
    print(f"[{tag}] start pos/vel/acc eval: {pos[0]} {vel0[0]} {acc0[0]}")
    print(f"[{tag}] start pos/vel/acc tgt : {np.asarray(start_pos)} {np.asarray(start_vel)} {np.asarray(start_acc)}")
    print(f"[{tag}] end   pos/vel/acc eval: {pos[-1]} {vel0[-1]} {acc0[-1]}")
    print(f"[{tag}] end   pos/vel/acc tgt : {np.asarray(end_pos)} {np.asarray(end_vel)} {np.asarray(end_acc)}")

def plot_trajectory(control_points: np.ndarray, traj: np.ndarray, title: str = "", save_path: Optional[Path] = None, show: bool = False):
    cp = np.asarray(control_points)
    tr = np.asarray(traj)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(tr[:, 0], tr[:, 1], "-", label="Trajectory (B-spline)")
    ax.plot(cp[:, 0], cp[:, 1], "rx--", label="Control Points")
    ax.scatter(cp[0, 0], cp[0, 1], c="green", marker="o", s=60, label="Start CP")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

# ----------------- Sampler 封装 -----------------

class StudentSampler:
    def __init__(self, ckpt_path: Optional[str] = None, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if ckpt_path is None:
            ckpt_path = Path(__file__).with_name("student_distilled.pth")
        self.ckpt_path = str(ckpt_path)

        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"Student checkpoint not found at: {self.ckpt_path}")

        ck = torch.load(self.ckpt_path, map_location="cpu")
        if isinstance(ck, dict) and "model" in ck:
            state = ck["model"]
        else:
            state = ck

        # Infer context_dim / hidden_dim / n_ctx_tokens / num_layers from state_dict.
        try:
            w0 = state["context_proj.0.weight"]  # (hidden_dim, context_dim)
        except KeyError as e:
            raise KeyError(
                "Checkpoint does not look like the current StudentTransformer. "
                "Expected key 'context_proj.0.weight'."
            ) from e

        inferred_hidden_dim = int(w0.shape[0])
        inferred_context_dim = int(w0.shape[1])

        # n_ctx_tokens from context_to_tokens weight: (n_ctx_tokens*hidden_dim, context_dim)
        w_tok = state.get("context_to_tokens.weight", None)
        if w_tok is None:
            raise KeyError("Missing 'context_to_tokens.weight' in checkpoint.")
        inferred_n_ctx_tokens = int(w_tok.shape[0] // inferred_hidden_dim)

        # num_layers from blocks.* keys
        block_ids = set()
        for k in state.keys():
            if k.startswith("blocks."):
                parts = k.split(".")
                if len(parts) >= 2 and parts[1].isdigit():
                    block_ids.add(int(parts[1]))
        inferred_num_layers = (max(block_ids) + 1) if block_ids else NUM_LAYERS

        self.context_dim = inferred_context_dim

        self.model = StudentTransformer(
            n_control_points=MODEL_HORIZON,
            state_dim=STATE_DIM,
            context_dim=inferred_context_dim,
            hidden_dim=inferred_hidden_dim,
            num_layers=inferred_num_layers,
            num_heads=NUM_HEADS,
            dropout=DROPOUT,
            n_ctx_tokens=inferred_n_ctx_tokens,
            cross_attn_layers=CROSS_ATTN_LAYERS,
        ).to(self.device)

        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def sample_control_points(
        self,
        task_mask: Sequence[float],
        start_pos: Sequence[float],
        start_vel: Sequence[float],
        start_acc: Sequence[float],
        yaw_vector: Sequence[float],
        geometry_offset: Sequence[float],
        guide_vector: Sequence[float],
        degree: int = BSPLINE_DEGREE,
        T_duration: float = T_DURATION,
    ) -> np.ndarray:
        """
        主接口：给定训练一致的 `context`（14 维）
        返回 remap 后的未归一化 full-horizon 控制点 (FULL_HORIZON,2)。
        """
        def _as_2d(x: Sequence[float] | np.ndarray | torch.Tensor) -> np.ndarray:
            arr = np.asarray(x, dtype=np.float32)
            if arr.ndim == 1:
                if arr.size != 2:
                    raise ValueError(f"Expected shape (2,), got {arr.shape}")
                return arr.reshape(1, 2)
            if arr.ndim == 2 and arr.shape[1] == 2:
                return arr
            raise ValueError(f"Expected shape (B,2), got {arr.shape}")

        # -------- 1) 按新的顺序接收参数并归一化，拼接为 context_normalized (B,14) --------
        # 输入顺序：tm, sp, sv, sa, yv, go, gv
        tm = torch.as_tensor(_as_2d(task_mask), device=self.device, dtype=torch.float32)
        sp_phys = torch.as_tensor(_as_2d(start_pos), device=self.device, dtype=torch.float32)
        sv_phys = torch.as_tensor(_as_2d(start_vel), device=self.device, dtype=torch.float32)
        sa_phys = torch.as_tensor(_as_2d(start_acc), device=self.device, dtype=torch.float32)
        yv = torch.as_tensor(_as_2d(yaw_vector), device=self.device, dtype=torch.float32)
        go_phys = torch.as_tensor(_as_2d(geometry_offset), device=self.device, dtype=torch.float32)
        gv_phys = torch.as_tensor(_as_2d(guide_vector), device=self.device, dtype=torch.float32)

        sp_norm = normalize_position_group_torch(sp_phys)
        sv_norm = normalize_velocity_group_torch(sv_phys)
        sa_norm = normalize_acceleration_group_torch(sa_phys)
        gv_norm = normalize_velocity_group_torch(gv_phys)
        go_norm = normalize_position_group_torch(go_phys)

        context = torch.cat([tm, sp_norm, sv_norm, sa_norm, yv, go_norm, gv_norm], dim=1)  # (B,14)

        # Student forward: 12-token normalized control points (B,12,2)
        tokens_norm = self.model(context)

        # -------- 2) 保留原始物理量用于 remap（不要归一化） --------
        context_d = {
            "task_mask": tm,
            "start_pos_phys": sp_phys,
            "start_vel_phys": sv_phys,
            "start_acc_phys": sa_phys,
            "guide_vector_phys": gv_phys,
            "geometry_offset_phys": go_phys,
        }

        # remap now returns (padded_phys (B, FULL_HORIZON, 2), lengths (B,))
        remap_out = remap_model_outputs_standalone(
            tokens_norm, FULL_HORIZON, context_d, degree=degree, T_duration=T_duration
        )
        if isinstance(remap_out, tuple) or isinstance(remap_out, list):
            full_phys_tensor, lengths = remap_out
        else:
            # backward compat: single tensor -> assume full length for all
            full_phys_tensor = remap_out
            lengths = torch.full((full_phys_tensor.shape[0],), FULL_HORIZON, dtype=torch.long, device=full_phys_tensor.device)

        full_phys_np = full_phys_tensor.detach().cpu().numpy()
        lengths_np = lengths.detach().cpu().numpy()

        # If single-sample, return (N,2) trimmed to meaningful length
        if full_phys_np.ndim == 3 and full_phys_np.shape[0] == 1:
            ln = int(lengths_np[0])
            return full_phys_np[0, :ln, :]
        # Multi-sample: return padded arrays plus lengths so caller can handle variable lengths
        return full_phys_np, lengths_np

    def sample_and_traj(
        self,
        task_mask: Sequence[float],
        start_pos: Sequence[float],
        start_vel: Sequence[float],
        start_acc: Sequence[float],
        yaw_vector: Sequence[float],
        geometry_offset: Sequence[float],
        guide_vector: Sequence[float],
        num_traj_points: int = 128,
        degree: int = BSPLINE_DEGREE,
        T_duration: float = T_DURATION,
    ):
        """
        返回 (control_points, traj) 两个 numpy 数组。
        """
        cps = self.sample_control_points(
            task_mask,
            start_pos,
            start_vel,
            start_acc,
            yaw_vector,
            geometry_offset,
            guide_vector,
            degree=degree,
            T_duration=T_duration,
        )

        print(f"cps: {cps}")

        # remap already returns the correct number of meaningful control points
        cps_for_traj = cps

        print("cps:", cps_for_traj)

        traj = bspline_trajectory(cps_for_traj, num_points=num_traj_points, degree=degree)
        return cps, traj

# ----------------- CLI Demo -----------------

def _demo():
    """
    简单 demo：随便造一个避免/碰撞 context，跑一遍并保存图。
    """
    sampler = StudentSampler()

    # 真实部署时必须使用与你训练/数据管线一致的 context（例如 14 维）。

    # 示例 1：避障任务 [0,1]
    task_mask = [0.0, 1.0]    # avoidance
    start_pos = [0.0, 0.0]
    start_vel = [0.5, 0.0]
    start_acc = [2.0, -1.0]    
    yaw_vector = [0.0, 0.0]    
    geometry_offset = [6.0, 4.0]
    guide_vector = [1.0, 0.0]

    cps_avoid, traj_avoid = sampler.sample_and_traj(
        task_mask, start_pos, start_vel, start_acc, yaw_vector, geometry_offset, guide_vector
    )
    plot_trajectory(cps_avoid, traj_avoid, title="Avoidance sample", save_path=Path("student_avoidance.png"))

    # 示例 2：碰撞任务 [1,0]
    task_mask_col = [1.0, 0.0]
    start_pos_coli = [0.0, 0.0]
    start_vel_coli = [3.70, 0.0]
    start_acc_coli = [-1.28, 0.0]
    yaw_vector_coli = [0.99, 0.03]
    geometry_offset_col = [4.02, 1.2]
    guide_vector_coli = [-0.65, 0.24]
    cps_col, traj_col = sampler.sample_and_traj(
        task_mask_col, start_pos_coli, start_vel_coli, start_acc_coli, yaw_vector_coli, geometry_offset_col, guide_vector_coli
    )
    plot_trajectory(cps_col, traj_col, title="Collision sample", save_path=Path("student_collision.png"))

    # 额外：针对碰撞轨迹，取前 16 个控制点（pre-collision）并绘制速度/加速度曲线
    try:
        ctrl_pre16 = cps_col[:16]
        # 估计 t_pre 与 vel_pre 的方法与 remap 中一致：
        collision_point = np.asarray(start_pos_coli) + np.asarray(geometry_offset_col)
        last_pred = cps_col[-1]
        L_post = np.linalg.norm(last_pred - collision_point)
        v_post_mag = np.linalg.norm(np.asarray(guide_vector_coli))
        eps = 1e-6
        t_post_val = 2.0 * L_post / (v_post_mag + eps)
        T_pre = float(T_DURATION - t_post_val)
        if T_pre <= 0:
            T_pre = float(max(0.1, T_DURATION * 0.5))

        print(f"Estimated T_pre={T_pre:.4f}s (t_post={t_post_val:.4f}s)")
        plot_vel_acc_for_control_points(
            ctrl_pre16,
            T_pre,
            num_points=256,
            degree=BSPLINE_DEGREE,
            save_path=Path("student_collision_vel_acc.png"),
        )
        print("Saved student_collision_vel_acc.png")
        # Also save finite-difference based velocity/acceleration for comparison
        try:
            plot_fd_vel_acc_for_control_points(
                ctrl_pre16,
                T_pre,
                num_points=256,
                degree=BSPLINE_DEGREE,
                save_path=Path("student_collision_vel_acc_fd.png"),
            )
            print("Saved student_collision_vel_acc_fd.png")
        except Exception as e_fd:
            print("Could not save finite-difference vel/acc plot:", e_fd)
    except Exception as e:
        print("Could not plot/enforce pre-collision vel/acc:", e)

    print("Saved student_avoidance.png and student_collision.png")

if __name__ == "__main__":
    _demo()