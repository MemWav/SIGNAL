import torch
import torch.nn as nn
from typing import List


class FrameTransformer(nn.Module):
    """
    한 프레임의 21개 (x,y,z)를 Transformer-encoder로 변환,
    선택적으로 multi-head attention map을 수집한다.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        return_attn: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.return_attn = return_attn
        self.attn_maps: List[torch.Tensor] = []

        # 21×3 → d_model
        self.proj = nn.Linear(3, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.randn(1, 22, d_model))  # 1 CLS + 21 landmarks

        def build_layer():
            return nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=4 * d_model,
                dropout=0.1,
                batch_first=True,
            )

        self.layers = nn.ModuleList([build_layer() for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

        # Hook 등록(선택)
        if self.return_attn:
            self._register_attn_hooks()

    # ------------------------------------------------------------------ #
    # Attention hook
    # ------------------------------------------------------------------ #
    def _register_attn_hooks(self) -> None:
        def capture_hook(m, _inp, out):
            """out = (attn_out, attn_weights)"""
            attn_weights = out[1]
            if attn_weights is not None:
                self.attn_maps.append(attn_weights.detach().cpu())

        for lyr in self.layers:
            lyr.self_attn.register_forward_hook(capture_hook)

    def _clear_attn(self) -> None:
        if self.return_attn:
            self.attn_maps.clear()

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        xyz: (B, 21, 3) → (B, d_model)  -- CLS feature
        """
        B = xyz.size(0)
        self._clear_attn()

        tok = self.proj(xyz)                               # (B, 21, d_model)
        x = torch.cat([self.cls_token.expand(B, -1, -1), tok], dim=1)
        x = x + self.pos                                   # 위치 임베딩

        for lyr in self.layers:                            # (B, 22, d_model)
            x = lyr(x)

        x = self.norm(x)
        return x[:, 0]                                     # (B, d_model)


class VideoGestureLSTM(nn.Module):
    """
    (B, T, 21, 3) landmark 시퀀스를 받아 제스처 클래스를 예측한다.
    - frame 인코딩: FrameTransformer
    - temporal 모델: 단/양방향 LSTM
    """

    def __init__(
        self,
        d_model: int = 128,
        lstm_hidden: int = 256,
        n_gestures: int = 9,
        return_attn: bool = False,
        bidirectional: bool = False,
        lstm_dropout: float = 0.2,
    ):
        super().__init__()
        self.frame_encoder = FrameTransformer(
            d_model=d_model,
            num_layers=4,
            num_heads=8,
            return_attn=return_attn,
        )

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.0,           # single-layer라 내부 dropout X
        )
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.post_norm = nn.LayerNorm(lstm_out_dim)
        self.post_drop = nn.Dropout(lstm_dropout)
        self.head = nn.Linear(lstm_out_dim, n_gestures)

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(self, video_xyz: torch.Tensor) -> torch.Tensor:
        """
        video_xyz: (B, T, 21, 3) → (B, n_gestures) logits
        """
        B, T, L, C = video_xyz.shape                       # L=21, C=3

        # ① 프레임 병렬 인코딩 (B*T, 21, 3)
        feats = self.frame_encoder(
            video_xyz.view(B * T, L, C)
        ).view(B, T, -1)                                   # (B, T, d_model)

        # ② LSTM
        lstm_out, _ = self.lstm(feats)                     # (B, T, H or 2H)

        # ③ 마지막 시점 + 정규화·드롭아웃
        last_hidden = lstm_out[:, -1, :]                   # (B, H or 2H)
        last_hidden = self.post_norm(last_hidden)
        last_hidden = self.post_drop(last_hidden)

        return self.head(last_hidden)                      # (B, n_gestures)
