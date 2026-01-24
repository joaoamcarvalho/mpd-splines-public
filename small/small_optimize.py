import math
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from small_deploy import (
  StudentSampler,
  BSPLINE_DEGREE,
  T_DURATION,
  plot_vel_acc_for_control_points,
  bspline_derivatives_np,
  _bspline_open_uniform_knots_torch,
  _bspline_deboor_eval_torch,
  bspline_trajectory,
  plot_trajectory,
)


def _bspline_derivative_ctrl_pts_torch(
  ctrl_pts: torch.Tensor,
  degree: int,
  knots: torch.Tensor,
  order: int = 1,
) -> tuple[torch.Tensor, int, torch.Tensor]:
  """Derivative control points for d^order C/du^order (Torch).

  ctrl_pts: (N,2)
  knots: (N + degree + 1,)
  returns: (ctrl_der, degree_der, knots_der)
  """
  if order < 0:
    raise ValueError(f"order must be >=0, got {order}")
  p = int(degree)
  if order == 0:
    return ctrl_pts, p, knots
  if order > p:
    # derivative above degree is zero curve; return single zero control point
    z = torch.zeros((1, ctrl_pts.shape[-1]), device=ctrl_pts.device, dtype=ctrl_pts.dtype)
    return z, 0, knots[order:-order]

  Pk = ctrl_pts
  Uk = knots
  pk = p
  for _ in range(order):
    # P'_{i} = pk * (P_{i+1}-P_{i}) / (U_{i+pk+1}-U_{i+1})
    denom = (Uk[pk + 1 : pk + 1 + (Pk.shape[0] - 1)] - Uk[1 : 1 + (Pk.shape[0] - 1)]).clamp(min=1e-12)
    Pk = (Pk[1:] - Pk[:-1]) * (float(pk) / denom.unsqueeze(-1))
    Uk = Uk[1:-1]
    pk -= 1
  return Pk, pk, Uk


def _bspline_vel_acc_samples_torch(
  ctrl_pts: torch.Tensor,
  degree: int,
  T: float,
  num_points: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Sample curve velocity/acceleration along time (Torch, differentiable).

  Returns vel, acc in time domain (m/s, m/s^2), shapes (M,2).
  """
  if num_points < 3:
    raise ValueError("num_points must be >= 3")
  N = int(ctrl_pts.shape[0])
  p = int(degree)
  if N <= p:
    z = torch.zeros((num_points, ctrl_pts.shape[-1]), device=ctrl_pts.device, dtype=ctrl_pts.dtype)
    return z, z

  T_eff = float(T) if float(T) > 1e-9 else 1e-9
  knots = _bspline_open_uniform_knots_torch(N, p, device=ctrl_pts.device, dtype=ctrl_pts.dtype)

  d1_ctrl, d1_deg, d1_knots = _bspline_derivative_ctrl_pts_torch(ctrl_pts, p, knots, order=1)
  d2_ctrl, d2_deg, d2_knots = _bspline_derivative_ctrl_pts_torch(ctrl_pts, p, knots, order=2)

  u = torch.linspace(0.0, 1.0, int(num_points), device=ctrl_pts.device, dtype=ctrl_pts.dtype)

  # Evaluate derivative curves with de Boor in batch: treat each u as a batch element.
  d1_ctrl_b = d1_ctrl.unsqueeze(0).expand(u.shape[0], -1, -1)
  d2_ctrl_b = d2_ctrl.unsqueeze(0).expand(u.shape[0], -1, -1)
  v_du = _bspline_deboor_eval_torch(d1_ctrl_b, u, d1_deg, d1_knots)
  a_du2 = _bspline_deboor_eval_torch(d2_ctrl_b, u, d2_deg, d2_knots)

  vel = v_du / T_eff
  acc = a_du2 / (T_eff * T_eff)
  return vel, acc


def _compute_costs_torch(
  q: torch.Tensor,
  max_vel: float,
  max_acc: float,
  degree: int,
  t_total: float,
  num_curve_samples: int,
  w_smooth: float,
  w_feas: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return (total_cost, smooth_cost, feas_cost) for control points q.

  q: (N,2)
  """
  device = q.device
  dtype = q.dtype
  n = q.shape[0]

  # Smoothness cost (jerk on control points)
  # Normalize by number of points to keep scale consistent
  if n >= 4:
    jerk = q[3:] - 3.0 * q[2:-1] + 3.0 * q[1:-2] - q[:-3]
    smooth_cost = (jerk.pow(2).sum(dim=1)).mean()
  else:
    smooth_cost = torch.zeros((), device=device, dtype=dtype)

  # Feasibility cost on the *curve* (true B-spline derivatives)
  vel, acc = _bspline_vel_acc_samples_torch(q, degree=degree, T=t_total, num_points=num_curve_samples)
  
  # Compute norms (magnitude)
  # Add epsilon to sqrt for gradient stability at 0 (though unlikely to be 0 for moving traj)
  v_norm = (vel.pow(2).sum(dim=-1) + 1e-8).sqrt()
  a_norm = (acc.pow(2).sum(dim=-1) + 1e-8).sqrt()

  # Penalize if norm > max
  # Using L2 penalty on the excess magnitude: mean( relu(|v| - vmax)^2 )
  # This has smaller gradients than relu(v^2 - vmax^2)^2
  v_excess = torch.relu(v_norm - max_vel)
  a_excess = torch.relu(a_norm - max_acc)
  
  feas_cost = v_excess.pow(2).mean() + a_excess.pow(2).mean()

  total_cost = (float(w_smooth) * smooth_cost) + (float(w_feas) * feas_cost)
  return total_cost, smooth_cost, feas_cost


def optimize_control_points_lbfgs(
  cps_init: np.ndarray,
  max_vel: float = 4.0,
  max_acc: float = 4.0,
  degree: int = BSPLINE_DEGREE,
  t_total: float = T_DURATION,
  max_iter: int = 200,
  lr: float = 0.1,
  num_curve_samples: int = 256,
  w_smooth: float = 1.0,
  w_feas: float = 100.0,
) -> np.ndarray:
  """Optimize control points using LBFGS with smoothness + feasibility costs.

  Only endpoint positions are kept fixed; other control points may change.
  Feasibility penalty is computed on the actual B-spline curve derivatives.
  """
  device = torch.device("cpu")
  q0 = torch.as_tensor(cps_init, dtype=torch.float32, device=device)
  n = q0.shape[0]

  if n <= 6:
    return cps_init.copy()

  # Optimize internal points (keep start/end pos, vel, acc fixed => fix first 3 and last 3 points)
  fixed_head = q0[0:3].detach()
  fixed_tail = q0[-3:].detach()
  internal_init = q0[3:-3].detach().clone()
  internal = torch.nn.Parameter(internal_init)

  # Use LBFGS with lower LR and strong_wolfe line search
  # Note: LBFGS can be unstable if gradients are massive. We rely on the cost formulation 
  # (relu on norm, mean reduction) to keep gradients reasonable.
  optimizer = torch.optim.LBFGS(
    [internal],
    lr=lr,
    max_iter=max_iter,
    history_size=20,
    line_search_fn="strong_wolfe",
  )

  def closure():
    optimizer.zero_grad(set_to_none=True)
    q = torch.cat([fixed_head, internal, fixed_tail], dim=0)
    total_cost, _, _ = _compute_costs_torch(
      q,
      max_vel=max_vel,
      max_acc=max_acc,
      degree=degree,
      t_total=t_total,
      num_curve_samples=num_curve_samples,
      w_smooth=w_smooth,
      w_feas=w_feas,
    )
    
    # Check for NaN
    if torch.isnan(total_cost):
       # If NaN, return a large value but don't backward to avoid corrupting internal state
       # (Though LBFGS might still fail, this is a safety check)
       return torch.tensor(1e6, requires_grad=True)

    total_cost.backward()
    
    # Optional: Gradient clipping if needed (LBFGS steps can be large)
    torch.nn.utils.clip_grad_norm_([internal], max_norm=10.0)
    
    return total_cost

  # Run optimization
  # Wrapped in try-except to catch LBFGS errors
  try:
      optimizer.step(closure)
  except Exception as e:
      print(f"Optimization failed or stopped early: {e}")

  q_opt = torch.cat([fixed_head, internal.detach(), fixed_tail], dim=0)
  return q_opt.cpu().numpy()


def run_demo():
  sampler = StudentSampler()

  # Collision task context (same shape as small_deploy demo)
  task_mask_col = [1.0, 0.0]
  start_pos_coli = [0.0, 0.0]
  start_vel_coli = [3.70, 0.0]
  start_acc_coli = [-1.28, 0.0]
  yaw_vector_coli = [0.99, 0.03]
  geometry_offset_col = [4.02, 1.2]
  guide_vector_coli = [-0.65, 0.24]

  cps_col = sampler.sample_control_points(
    task_mask_col,
    start_pos_coli,
    start_vel_coli,
    start_acc_coli,
    yaw_vector_coli,
    geometry_offset_col,
    guide_vector_coli,
    degree=BSPLINE_DEGREE,
    T_duration=T_DURATION,
  )

  # Pre-collision trajectory uses 16 control points.
  cps_col = np.asarray(cps_col, dtype=np.float32)
  if cps_col.ndim != 2 or cps_col.shape[1] != 2:
    raise ValueError(f"Expected control points (N,2), got {cps_col.shape}")
  if cps_col.shape[0] < 16:
    raise ValueError(f"Expected at least 16 pre-collision control points, got N={cps_col.shape[0]}")
  cps_pre = cps_col[:16].copy()

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

  # Optimize
  cps_opt = optimize_control_points_lbfgs(
    cps_pre,
    max_vel=4.0,
    max_acc=4.0,
    degree=BSPLINE_DEGREE,
    t_total=T_pre,
    max_iter=500,
    lr=0.1,
    num_curve_samples=256,
    w_smooth=0.1,
    w_feas=1000.0,
  )

  # Print max stats using the same deploy derivative function used for plotting.
  _, vel_b, acc_b, _ = bspline_derivatives_np(cps_pre, T_pre, num_points=512, degree=BSPLINE_DEGREE)
  _, vel_a, acc_a, _ = bspline_derivatives_np(cps_opt, T_pre, num_points=512, degree=BSPLINE_DEGREE)
  print(
    "Before: max|v|=%.3f max|a|=%.3f |  After: max|v|=%.3f max|a|=%.3f"
    % (
      float(np.linalg.norm(vel_b, axis=1).max()),
      float(np.linalg.norm(acc_b, axis=1).max()),
      float(np.linalg.norm(vel_a, axis=1).max()),
      float(np.linalg.norm(acc_a, axis=1).max()),
    )
  )

  # Plot velocity/acceleration curves (before & after)
  out_dir = Path("viz_opt")
  out_dir.mkdir(exist_ok=True)

  before_vel_path = out_dir / "opt_before_vel_acc.png"
  after_vel_path = out_dir / "opt_after_vel_acc.png"
  plot_vel_acc_for_control_points(
    cps_pre,
    T_pre,
    num_points=256,
    degree=BSPLINE_DEGREE,
    save_path=before_vel_path,
  )
  plot_vel_acc_for_control_points(
    cps_opt,
    T_pre,
    num_points=256,
    degree=BSPLINE_DEGREE,
    save_path=after_vel_path,
  )

  # Also plot B-spline spatial trajectories (before & after)
  traj_before = bspline_trajectory(cps_pre, num_points=512, degree=BSPLINE_DEGREE)
  traj_after = bspline_trajectory(cps_opt, num_points=512, degree=BSPLINE_DEGREE)

  traj_before_path = out_dir / "traj_before.png"
  traj_after_path = out_dir / "traj_after.png"
  plot_trajectory(cps_pre, traj_before, title="B-spline Trajectory Before", save_path=traj_before_path)
  plot_trajectory(cps_opt, traj_after, title="B-spline Trajectory After", save_path=traj_after_path)

  print(f"Saved {before_vel_path} {after_vel_path} {traj_before_path} {traj_after_path}")


if __name__ == "__main__":
  run_demo()
