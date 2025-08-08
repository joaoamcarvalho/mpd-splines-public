import einops
import torch
import torch.nn as nn

from mpd.models.layers.layers import (
    Downsample1d,
    Conv1dBlock,
    Upsample1d,
    ResidualTemporalBlock,
    TimeEncoder,
    MLP,
    group_norm_n_groups,
    LinearAttention,
    PreNorm,
    Residual,
)
from mpd.models.layers.layers_attention import SpatialTransformer
from torch_robotics.torch_utils.torch_timer import TimerCUDA

import numpy as np

UNET_DIM_MULTS = {
    0: (1, 2, 4),
    1: (1, 2, 4, 8),
    2: (1, 2),
}


class TemporalUnet(nn.Module):

    def __init__(
        self,
        n_support_points=None,
        state_dim=None,
        unet_input_dim=32,
        dim_mults=(1, 2, 4, 8),
        time_emb_dim=32,
        self_attention=False,
        conditioning_embed_dim=4,
        conditioning_type=None,
        attention_num_heads=2,
        attention_dim_head=32,
        **kwargs,
    ):
        super().__init__()

        self.n_support_points = n_support_points
        self.state_dim = state_dim
        input_dim = state_dim

        # Conditioning
        if conditioning_type is None or conditioning_type == "None":
            conditioning_type = None
        elif conditioning_type == "concatenate":
            if self.state_dim < conditioning_embed_dim // 4:
                # Embed the state in a latent space HxF if the conditioning embedding is much larger than the state
                state_emb_dim = conditioning_embed_dim // 4
                self.state_encoder = MLP(state_dim, state_emb_dim, hidden_dim=state_emb_dim, n_layers=2, act="mish")
            else:
                state_emb_dim = state_dim
                self.state_encoder = nn.Identity()
            input_dim = state_emb_dim + conditioning_embed_dim
        elif conditioning_type == "attention":
            pass
        elif conditioning_type == "default":
            pass
        else:
            raise NotImplementedError
        self.conditioning_type = conditioning_type

        dims = [input_dim, *map(lambda m: unet_input_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f"[ models/temporal ] Channel dimensions: {in_out}")

        # Networks
        self.time_mlp = TimeEncoder(32, time_emb_dim)

        # conditioning dimension (time + context)
        cond_dim = time_emb_dim + (conditioning_embed_dim if conditioning_type == "default" else 0)

        # Unet
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(dim_in, dim_out, cond_dim, n_support_points=n_support_points),
                        ResidualTemporalBlock(dim_out, dim_out, cond_dim, n_support_points=n_support_points),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))) if self_attention else nn.Identity(),
                        (
                            SpatialTransformer(
                                dim_out,
                                attention_num_heads,
                                attention_dim_head,
                                depth=1,
                                context_dim=conditioning_embed_dim,
                            )
                            if conditioning_type == "attention"
                            else None
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

            if not is_last:
                n_support_points = n_support_points // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, cond_dim, n_support_points=n_support_points)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if self_attention else nn.Identity()
        self.mid_attention = (
            SpatialTransformer(
                mid_dim, attention_num_heads, attention_dim_head, depth=1, context_dim=conditioning_embed_dim
            )
            if conditioning_type == "attention"
            else nn.Identity()
        )
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, cond_dim, n_support_points=n_support_points)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(dim_out * 2, dim_in, cond_dim, n_support_points=n_support_points),
                        ResidualTemporalBlock(dim_in, dim_in, cond_dim, n_support_points=n_support_points),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))) if self_attention else nn.Identity(),
                        (
                            SpatialTransformer(
                                dim_in,
                                attention_num_heads,
                                attention_dim_head,
                                depth=1,
                                context_dim=conditioning_embed_dim,
                            )
                            if conditioning_type == "attention"
                            else None
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

            if not is_last:
                n_support_points = n_support_points * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(unet_input_dim, unet_input_dim, kernel_size=5, n_groups=group_norm_n_groups(unet_input_dim)),
            nn.Conv1d(unet_input_dim, state_dim, 1),
        )

    def forward(self, x, time, context):
        """
        x : [ batch x horizon x state_dim ]
        context: [batch x context_dim]
        """
        b, h, d = x.shape

        t_emb = self.time_mlp(time)
        c_emb = t_emb
        if self.conditioning_type == "concatenate":
            x_emb = self.state_encoder(x)
            context = einops.repeat(context, "m n -> m h n", h=h)
            x = torch.cat((x_emb, context), dim=-1)
        elif self.conditioning_type == "attention":
            # reshape to keep the interface
            context = einops.rearrange(context, "b d -> b 1 d")
        elif self.conditioning_type == "default":
            c_emb = torch.cat((t_emb, context), dim=-1)

        # swap horizon and channels (state_dim)
        x = einops.rearrange(x, "b h c -> b c h")  # batch, horizon, channels (state_dim)

        h = []
        for resnet, resnet2, attn_self, attn_conditioning, downsample in self.downs:
            x = resnet(x, c_emb)
            # if self.conditioning_type == 'attention':
            #     x = attention1(x, context=conditioning_emb)
            x = resnet2(x, c_emb)
            x = attn_self(x)
            if self.conditioning_type == "attention":
                x = attn_conditioning(x, context=context)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, c_emb)
        x = self.mid_attn(x)
        if self.conditioning_type == "attention":
            x = self.mid_attention(x, context=context)
        x = self.mid_block2(x, c_emb)

        for resnet, resnet2, attn_self, attn_conditioning, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, c_emb)
            x = resnet2(x, c_emb)
            x = attn_self(x)
            if self.conditioning_type == "attention":
                x = attn_conditioning(x, context=context)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, "b c h -> b h c")

        return x


if __name__ == "__main__":
    import torch
    import time

    device = "cuda:0"

    batch_size = 1000
    n_support_points = 16

    model = TemporalUnet(
        n_support_points=n_support_points,
        state_dim=7,
        unet_input_dim=32,
        dim_mults=UNET_DIM_MULTS[1],
        time_emb_dim=32,
        self_attention=True,
        conditioning_embed_dim=128,
        conditioning_type="default",
    )
    model.to(device)

    x = torch.randn(batch_size, n_support_points, 7, device=device)  # batch_size x horizon x state_dim
    t = torch.randn(batch_size, device=device)  # batch_size x horizon
    context = torch.randn(batch_size, 128, device=device)  # batch_size x context_dim

    t_elapsed_l = []
    for i in range(20):
        with TimerCUDA() as t_forward:
            output = model(x, t, context)
        # print("Time taken (CUDA):", t_forward.elapsed)
        t_elapsed_l.append(t_forward.elapsed)
    print("Time taken average (CUDA):", np.mean(t_elapsed_l[10:]))
