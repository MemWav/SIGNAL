import torch
import torch.nn as nn

class HandGestureTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        n_gestures: int = 4,
        return_attn: bool = False,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(3, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.randn(1, 22, d_model))
        self.return_attn = return_attn
        self.attn_maps: list[torch.Tensor] = []

        def build_layer():
            return nn.TransformerEncoderLayer(
                d_model, num_heads, 4 * d_model, dropout=0.1, batch_first=True
            )

        self.layers = nn.ModuleList([build_layer() for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_gestures)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        b = xyz.size(0)
        tok = self.proj(xyz)                         # (B,21,d)
        x = torch.cat([self.cls_token.expand(b, -1, -1), tok], dim=1) + self.pos
        self.attn_maps.clear()

        for layer in self.layers:
            if self.return_attn:
                def _hook(mod, _inp, _out):
                    self.attn_maps.append(mod.self_attn.attn_output_weights.detach().cpu())
                h = layer.self_attn.register_forward_hook(_hook)
                x = layer(x)
                h.remove()
            else:
                x = layer(x)

        x = self.norm(x)
        return self.head(x[:, 0])                    # (B, n_gestures)
