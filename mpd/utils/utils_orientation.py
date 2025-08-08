# From  https://github.com/qazwsxal/diffusion-extensions/blob/master/util.py
# Appendix B of https://arxiv.org/pdf/1812.07035.pdf
import torch


def rmat2six(x: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to six parameters.
    """
    # Drop last column
    return torch.flatten(x[..., :2, :], -2, -1)


def six2rmat(x: torch.Tensor) -> torch.Tensor:
    """
    Convert six parameters to rotation matrix.
    """
    a1 = x[..., :3]
    a2 = x[..., 3:6]
    b1 = a1 / a1.norm(p=2, dim=-1, keepdim=True)
    b1_a2 = (b1 * a2).sum(dim=-1, keepdim=True)  # Dot product
    b2 = a2 - b1_a2 * b1
    b2 = b2 / b2.norm(p=2, dim=-1, keepdim=True)
    b3 = torch.cross(b1, b2, dim=-1)
    out = torch.stack((b1, b2, b3), dim=-2)
    return out
