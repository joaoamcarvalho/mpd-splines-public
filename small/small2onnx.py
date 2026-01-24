#!/usr/bin/env python3
"""
Export small/student_distilled.pth -> ONNX with dynamic batch dim.

Hardcoded:
  CKPT_PATH = "small/student_distilled.pth"
  OUT_ONNX   = "small/student_distilled.onnx"
  MIN_BATCH  = 1
  OPT_BATCH  = 144
  MAX_BATCH  = 144

This wrapper accepts raw context per sample:
  [start_pos(2), start_vel(2), start_acc(2), guide_vector(2), yaw_vector(2), task_mask(2), geometry_offset(2)]
-> shape (N,14)

Inside the wrapper we apply the same normalization rules used in the student code,
runs StudentTransformer, then denormalizes control points by POS_STD.
This script replaces nn.MultiheadAttention with an ONNX-exportable implementation.
"""
import os
import sys
import torch
import numpy as np
import onnx
import torch.nn as nn
import math
import torch.nn.functional as F
# Import StudentTransformer definition
from small.small_deploy import StudentTransformer

CKPT_PATH = "small/student_distilled.pth"
OUT_ONNX = "small/student_distilled.onnx"
MIN_BATCH = 1
OPT_BATCH = 72
MAX_BATCH = 72
OPSET = 18

# Normalization constants copied from small/small_deploy.py
POS_STD = 2.4662630558013916 #1.820559 
VEL_STD = 1.5834906101226807 #0.842956
ACC_STD = 2.285597085952759 #1.718903

# ---------------- Model constants ----------------
# StudentTransformer outputs 12 control points (MODEL_HORIZON).
MODEL_HORIZON = 12

STATE_DIM = 2
CONTEXT_DIM = 14

# These are not fully inferable from the checkpoint, but must match training.
NUM_HEADS = 8
DROPOUT = 0.1
N_CTX_TOKENS = 4
CROSS_ATTN_LAYERS = 2

# ---------------- Exportable Multi-Head Attention ----------------
class ExportableMHA(nn.Module):
    def __init__(self, orig_mha: nn.MultiheadAttention, batch_first: bool = True):
        super().__init__()
        embed_dim = orig_mha.embed_dim
        num_heads = orig_mha.num_heads
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # 定义投影层
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        # === 复制权重 (核心逻辑) ===
        # 尝试从 in_proj 复制
        copied = False
        if hasattr(orig_mha, "in_proj_weight") and orig_mha.in_proj_weight is not None:
            try:
                # 复制到合并的 in_proj
                self.in_proj.weight.data.copy_(orig_mha.in_proj_weight.data)
                if orig_mha.in_proj_bias is not None:
                    self.in_proj.bias.data.copy_(orig_mha.in_proj_bias.data)
                else:
                    self.in_proj.bias.data.zero_()
                
                # 同时复制到分离的 proj (作为 fallback)
                w = orig_mha.in_proj_weight.data
                b = orig_mha.in_proj_bias.data if orig_mha.in_proj_bias is not None else torch.zeros_like(w[:,0])
                self.q_proj.weight.data.copy_(w[:embed_dim])
                self.q_proj.bias.data.copy_(b[:embed_dim])
                self.k_proj.weight.data.copy_(w[embed_dim:2*embed_dim])
                self.k_proj.bias.data.copy_(b[embed_dim:2*embed_dim])
                self.v_proj.weight.data.copy_(w[2*embed_dim:])
                self.v_proj.bias.data.copy_(b[2*embed_dim:])
                copied = True
            except Exception:
                pass
        
        # 如果从 in_proj 复制失败，尝试从 q_proj/k_proj/v_proj 复制
        if not copied:
            try:
                self.q_proj.weight.data.copy_(orig_mha.q_proj.weight.data)
                self.q_proj.bias.data.copy_(orig_mha.q_proj.bias.data)
                self.k_proj.weight.data.copy_(orig_mha.k_proj.weight.data)
                self.k_proj.bias.data.copy_(orig_mha.k_proj.bias.data)
                self.v_proj.weight.data.copy_(orig_mha.v_proj.weight.data)
                self.v_proj.bias.data.copy_(orig_mha.v_proj.bias.data)
            except Exception:
                pass # 保持随机初始化（警告：这在实际部署会出错，但在 debug 脚本中可能忽略）

        # 复制 out_proj
        try:
            self.out_proj.weight.data.copy_(orig_mha.out_proj.weight.data)
            if orig_mha.out_proj.bias is not None:
                self.out_proj.bias.data.copy_(orig_mha.out_proj.bias.data)
        except Exception:
            pass

    def forward(self, query, key, value, need_weights: bool = False):
        # 1. 投影
        use_in_proj = (query is key) and (query is value)
        if use_in_proj:
            qkv = self.in_proj(query)
            Q, K, V = qkv.chunk(3, dim=-1)
        else:
            Q = self.q_proj(query)
            K = self.k_proj(key)
            V = self.v_proj(value)

        # 2. 调整形状 (Batch, SeqLen, Heads, HeadDim) -> (Batch, Heads, SeqLen, HeadDim)
        B, Lq, _ = Q.shape
        B, Lk, _ = K.shape
        B, Lv, _ = V.shape
        
        Q = Q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Lv, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. 使用 PyTorch 原生 SDPA (导出为高效算子)
        # 这一步是提速的关键！
        x = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V, 
            attn_mask=None, 
            dropout_p=0.0, 
            is_causal=False
        )

        # 4. 恢复形状
        x = x.transpose(1, 2).contiguous().view(B, Lq, self.embed_dim)
        
        # 5. 输出投影
        return self.out_proj(x), None

# ---------------- Export wrapper ----------------
class ExportWrapper(nn.Module):
    """
            Accepts raw context: (B,14) per row in the new order:
                [task_mask(2), start_pos(2), start_vel(2), start_acc(2), yaw_vector(2), geometry_offset(2), guide_vector(2)]

        ONNX only does (vectorized):
            1) Normalize inputs (same rule as small/small_deploy.py)
            2) Run StudentTransformer -> 12 normalized control points
            3) Denormalize control points to physical units by POS_STD

        NOTE: remap/trajectory assembly is intentionally done outside ONNX.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        # register constants as buffers to embed them in the graph
        self.register_buffer("pos_std", torch.tensor(POS_STD, dtype=torch.float32))
        self.register_buffer("vel_std", torch.tensor(VEL_STD, dtype=torch.float32))
        self.register_buffer("acc_std", torch.tensor(ACC_STD, dtype=torch.float32))

    def normalize_ctx(self, x_raw: torch.Tensor) -> torch.Tensor:
        # x_raw: (N,14)
        # new fields order: tm(0:2), sp(2:4), sv(4:6), sa(6:8), yv(8:10), go(10:12), gv(12:14)
        tm = x_raw[:, 0:2]
        sp = x_raw[:, 2:4]
        sv = x_raw[:, 4:6]
        sa = x_raw[:, 6:8]
        yv = x_raw[:, 8:10]
        go = x_raw[:, 10:12]
        gv = x_raw[:, 12:14]

        sp_n = sp / self.pos_std
        sv_n = sv / self.vel_std
        sa_n = sa / self.acc_std
        # yaw and task mask remain unchanged
        go_n = go / self.pos_std
        gv_n = gv / self.vel_std

        # return in model-expected order: [tm, sp_n, sv_n, sa_n, yv, go_n, gv_n]
        return torch.cat([tm, sp_n, sv_n, sa_n, yv, go_n, gv_n], dim=1)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        # x_raw: (B,14) in physical units
        x_raw = x_raw.to(next(self.model.parameters()).device)

        # normalize context and run student
        ctx_n = self.normalize_ctx(x_raw)
        cp_norm = self.model(ctx_n)  # (B,12,2) normalized control points
        cp_phys = cp_norm * self.pos_std  # (B,12,2) physical control points
        return cp_phys


# ---------------- utilities ----------------
def load_state_dict_from_ckpt(path):
    state = torch.load(path, map_location="cpu")
    # if checkpoint is wrapped
    if isinstance(state, dict):
        for k in ("model", "state_dict", "net"):
            if k in state:
                state = state[k]
                break
    return state


def replace_mha_in_model(model: nn.Module):
    """
    Walk the StudentTransformer blocks and replace nn.MultiheadAttention with ExportableMHA.
    """
    replaced = 0
    for blk in getattr(model, "blocks", []):
        # replace self_attn
        if hasattr(blk, "self_attn") and isinstance(blk.self_attn, nn.MultiheadAttention):
            orig = blk.self_attn
            blk.self_attn = ExportableMHA(orig, batch_first=True)
            replaced += 1
        # replace cross_attn if present
        if hasattr(blk, "cross_attn") and isinstance(blk.cross_attn, nn.MultiheadAttention):
            orig = blk.cross_attn
            blk.cross_attn = ExportableMHA(orig, batch_first=True)
            replaced += 1
    return replaced

def main():
    if not os.path.exists(CKPT_PATH):
        print(f"Checkpoint not found: {CKPT_PATH}", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cpu")

    sd = load_state_dict_from_ckpt(CKPT_PATH)

    # Infer key dims from checkpoint (hidden_dim/context_dim/n_ctx_tokens/num_layers)
    w0 = sd["context_proj.0.weight"]  # (hidden_dim, context_dim)
    inferred_hidden_dim = int(w0.shape[0])
    inferred_context_dim = int(w0.shape[1])

    w_tok = sd["context_to_tokens.weight"]  # (n_ctx_tokens*hidden_dim, context_dim)
    inferred_n_ctx_tokens = int(w_tok.shape[0] // inferred_hidden_dim)

    block_ids = set()
    for k in sd.keys():
        if k.startswith("blocks."):
            parts = k.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                block_ids.add(int(parts[1]))
    inferred_num_layers = (max(block_ids) + 1) if block_ids else 4

    # create model with inferred dims + fixed heads (must match training)
    model = StudentTransformer(
        n_control_points=MODEL_HORIZON,
        state_dim=STATE_DIM,
        context_dim=inferred_context_dim,
        hidden_dim=inferred_hidden_dim,
        num_layers=inferred_num_layers,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        n_ctx_tokens=inferred_n_ctx_tokens,
        cross_attn_layers=CROSS_ATTN_LAYERS,
    ).to(device)

    model.load_state_dict(sd, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Replace nn.MultiheadAttention with exportable implementation
    n_replaced = replace_mha_in_model(model)
    # n_ln = replace_layernorm_with_fp32(model)
    print(f"[small2onnx] Replaced {n_replaced} MultiheadAttention modules with ExportableMHA (using SDPA)")

    # wrap with normalization + denorm
    wrapper = ExportWrapper(model).to(device)
    wrapper.eval()

    example_in = torch.zeros((OPT_BATCH, CONTEXT_DIM), dtype=torch.float32, device=device)

    input_names = ["inputs"]
    output_names = ["control_points"]
    dynamic_axes = {
        "inputs": {0: "batch_size"},
        "control_points": {0: "batch_size"},
    }

    print(f"Exporting ONNX -> {OUT_ONNX} (opset {OPSET}), example batch={OPT_BATCH}, input_dim={CONTEXT_DIM}")
    torch.onnx.export(
        wrapper,
        example_in,
        OUT_ONNX,
        opset_version=OPSET,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        verbose=False,
    )

    # sanity check
    try:
        onnx_model = onnx.load(OUT_ONNX)
        onnx.checker.check_model(onnx_model)
        # materialize static shape info (helps some runtimes pick up dim=22)
        try:
            from onnx import shape_inference
            onnx_model = shape_inference.infer_shapes(onnx_model)
            onnx.save(onnx_model, OUT_ONNX)
        except Exception:
            pass
        print("ONNX model check: OK")
    except Exception as e:
        print("Warning: ONNX check failed:", e)

    print("Saved ONNX:", OUT_ONNX)
    print("onnxsim student_distilled.onnx student_distilled_sim.onnx --simplify")
    print()
    print("TensorRT build example (using trtexec):")
    print(
        "trtexec --onnx={} "
        "--minShapes=inputs:{}x{} --optShapes=inputs:{}x{} --maxShapes=inputs:{}x{} "
        "--saveEngine=small/student_distilled.engine --memPoolSize=workspace:512M --fp16".format(
            OUT_ONNX, MIN_BATCH, CONTEXT_DIM, OPT_BATCH, CONTEXT_DIM, MAX_BATCH, CONTEXT_DIM
        )
    )
    print()
    print("Note: deploy C++ should now pass raw 14-float context rows per sample.")


if __name__ == "__main__":
    main()