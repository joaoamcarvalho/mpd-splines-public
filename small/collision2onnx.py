#!/usr/bin/env python3
"""
Export small/best_model_multimat.pth -> ONNX with dynamic batch dim.

Hardcoded:
  CKPT_PATH = "small/best_model_multimat.pth"
  OUT_ONNX   = "small/best_model_multimat.onnx"
  MIN_BATCH  = 1
  OPT_BATCH  = 72
  MAX_BATCH  = 72

This script exports a model whose input is raw physical data per sample:
  [state(6), quat(4) in (w,x,y,z), phys(2)]  -> shape (N,12)

The wrapper converts quat->rot6d, normalizes using hardcoded statistics,
runs the ResMLP, then denormalizes the outputs. So the deployed C++ only
needs to memcpy raw (12 floats/sample) to the device.

This version performs a light "export-side fusion": it replaces encoder's
Linear+GELU and each ResidualBlock with fused module wrappers that keep the
same parameters but present fewer module boundaries during ONNX export.
"""
import os
import sys
import numpy as np
import torch
import onnx
import torch.nn as nn

# Import model definition from the repo
from small.collision_predict import ResMLP, ResidualBlock

CKPT_PATH = "small/best_model_multimat.pth"
OUT_ONNX = "small/best_model_multimat.onnx"
MIN_BATCH = 1
OPT_BATCH = 36
MAX_BATCH = 36
OPSET = 13

# Hardcoded statistics (copied from Python model file)
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


class ExportWrapper(torch.nn.Module):
    """
    Wrapper that accepts raw inputs (N,12): [state6, quat4 (w,x,y,z), phys2]
    and produces denormalized physical outputs (N,6).
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        # register statistics as buffers so they become constants in the graph
        self.register_buffer("state_mean", torch.from_numpy(STATE_MEAN))
        self.register_buffer("state_std", torch.from_numpy(STATE_STD))
        self.register_buffer("phys_mean", torch.from_numpy(PHYS_MEAN))
        self.register_buffer("phys_std", torch.from_numpy(PHYS_STD))
        self.register_buffer("target_mean", torch.from_numpy(TARGET_MEAN))
        self.register_buffer("target_std", torch.from_numpy(TARGET_STD))

    def quat_to_6d(self, q: torch.Tensor) -> torch.Tensor:
        # q: (N,4) with (w,x,y,z)
        w = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]

        # normalize quaternion
        norm = torch.sqrt(w * w + x * x + y * y + z * z)
        norm = torch.clamp(norm, min=1e-8)
        w = w / norm
        x = x / norm
        y = y / norm
        z = z / norm

        # compute rotation matrix entries (vectorized)
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        R00 = 1.0 - 2.0 * (yy + zz)
        R01 = 2.0 * (xy - wz)
        R02 = 2.0 * (xz + wy)

        R10 = 2.0 * (xy + wz)
        R11 = 1.0 - 2.0 * (xx + zz)
        R12 = 2.0 * (yz - wx)

        R20 = 2.0 * (xz - wy)
        R21 = 2.0 * (yz + wx)
        R22 = 1.0 - 2.0 * (xx + yy)

        # r6d = [R[:,0], R[:,1]] => [R00,R10,R20, R01,R11,R21]
        r6d = torch.stack([R00, R10, R20, R01, R11, R21], dim=1)
        return r6d

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        # x_raw: (N,12): [state(6), quat(4) (w,x,y,z), phys(2)]
        x_raw = x_raw.to(self.state_mean.device)
        state = x_raw[:, :6]
        quat = x_raw[:, 6:10]
        phys = x_raw[:, 10:12]

        rot6d = self.quat_to_6d(quat)  # (N,6)

        # normalize
        state_norm = (state - self.state_mean) / self.state_std
        phys_norm = (phys - self.phys_mean) / self.phys_std

        x_norm = torch.cat([state_norm, rot6d, phys_norm], dim=1)  # (N,14)
        y_norm = self.model(x_norm)                                # (N,6)
        y_phys = y_norm * self.target_std + self.target_mean       # denorm inside model
        return y_phys


# ------------------ Export-side fused modules ------------------

class FusedLinearGELU(nn.Module):
    """
    Wrapper that fuses a Linear followed by GELU into a single module (keeps the same Linear weights).
    This does not change math; it reduces module boundaries for export.
    """
    def __init__(self, linear: nn.Linear):
        super().__init__()
        # reuse the original Linear (weights & bias)
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.linear(x))


class FusedResidualBlock(nn.Module):
    """
    Wrapper that recreates the ResidualBlock computation inside one module.
    It reuses the original submodules (linears and layernorms) so parameters are preserved.
    Forward implements:
      t = linear1(x); t = ln1(t); t = GELU(t);
      t = linear2(t); t = ln2(t);
      return GELU(x + t)
    This reduces the number of module call sites during export.
    """
    def __init__(self, block: ResidualBlock):
        super().__init__()
        # block.block is nn.Sequential([Linear, LayerNorm, GELU, Linear, LayerNorm])
        seq = block.block
        # reuse original modules so parameters remain identical
        self.linear1 = seq[0]
        self.ln1 = seq[1]
        # seq[2] is GELU functional; we'll call F.gelu in forward
        self.linear2 = seq[3]
        self.ln2 = seq[4]
        # final activation
        self.activation = block.activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.linear1(x)
        t = self.ln1(t)
        t = torch.nn.functional.gelu(t)
        t = self.linear2(t)
        t = self.ln2(t)
        return torch.nn.functional.gelu(x + t)


def fuse_model(model: ResMLP):
    """
    Replace encoder Linear+GELU and each ResidualBlock with fused wrappers.
    This operates in-place on the model and returns it.
    """
    # fuse encoder if it's Sequential of [Linear, GELU]
    try:
        if isinstance(model.encoder, nn.Sequential) and len(model.encoder) >= 1 and isinstance(model.encoder[0], nn.Linear):
            lin = model.encoder[0]
            # replace encoder with fused module (keeps original linear)
            model.encoder = FusedLinearGELU(lin)
    except Exception as e:
        print("[fuse_model] encoder fuse skipped:", e)

    # replace each residual block
    for i, blk in enumerate(model.res_blocks):
        if isinstance(blk, ResidualBlock):
            model.res_blocks[i] = FusedResidualBlock(blk)
    return model


# ------------------ utilities ------------------

def load_state_dict_from_ckpt(path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        for k in ("model", "state_dict", "net"):
            if k in state:
                state = state[k]
                break
    return state


def main():
    if not os.path.exists(CKPT_PATH):
        print(f"Checkpoint not found: {CKPT_PATH}", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cpu")
    model = ResMLP(input_dim=14, output_dim=6, hidden_dim=1024, num_layers=6).to(device)
    sd = load_state_dict_from_ckpt(CKPT_PATH)
    model.load_state_dict(sd)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # ---- fuse model on export-side ----
    model = fuse_model(model)

    wrapper = ExportWrapper(model).to(device)
    wrapper.eval()

    # example input uses opt batch for export (now input dim = 12)
    example_in = torch.zeros((OPT_BATCH, 12), dtype=torch.float32, device=device)

    input_names = ["inputs"]
    output_names = ["outputs"]
    dynamic_axes = {
        "inputs": {0: "batch_size"},
        "outputs": {0: "batch_size"},
    }

    print(f"Exporting ONNX -> {OUT_ONNX} (opset {OPSET}), example batch={OPT_BATCH}, input_dim=12")
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

    # sanity check using onnx
    try:
        onnx_model = onnx.load(OUT_ONNX)
        onnx.checker.check_model(onnx_model)
        print("ONNX model check: OK")
    except Exception as e:
        print("Warning: ONNX check failed:", e)

    print("Saved ONNX:", OUT_ONNX)
    print()
    print("TensorRT build example (using trtexec):")
    print("trtexec --onnx={} --minShapes=inputs:{}x12 --optShapes=inputs:{}x12 --maxShapes=inputs:{}x12 --saveEngine=small/best_model_multimat.engine --memPoolSize=workspace:64M --fp16".format(
        OUT_ONNX, MIN_BATCH, OPT_BATCH, MAX_BATCH
    ))
    print()
    print("Note: deploy C++ should now pass raw 12-float samples per row: [state6, quat4(w,x,y,z), phys2]")

if __name__ == "__main__":
    main()