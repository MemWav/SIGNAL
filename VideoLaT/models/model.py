import torch.nn as nn
import torch

class EncoderLayerWithAttn(nn.TransformerEncoderLayer):
    """
    forward(src, need_weights=False, **kwargs)
    need_weights=True → (out, attn)  튜플 반환
    """
    def forward(self, src, need_weights=False, **kwargs):
        # ---- Self-Attention ----
        attn_out, attn_w = self.self_attn(
            src, src, src,
            need_weights=need_weights,          # 핵심!
            average_attn_weights=False,         # head별 그대로
            **kwargs
        )
        src = src + self.dropout1(attn_out)
        src = self.norm1(src)

        # ---- FFN ----
        ffn_out = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ffn_out)
        src = self.norm2(src)

        return (src, attn_w) if need_weights else src

class FrameEncoder(nn.Module):
    def __init__(self, d_model=128, n_heads=8, n_layers=4, return_attn=False):
        super().__init__()
        self.return_attn = return_attn
        self.joint_proj = nn.Linear(3, d_model)
        self.cls_token  = nn.Parameter(torch.zeros(1,1,d_model))
        self.pos_embed  = nn.Parameter(torch.randn(1,22,d_model))

        self.layers = nn.ModuleList([
            EncoderLayerWithAttn(d_model, n_heads, dim_feedforward=4*d_model, batch_first=True)
            for _ in range(n_layers)
        ])

    def forward(self, joints):                # (B,21,3)
        B = joints.size(0)
        tok = self.joint_proj(joints)         # (B,21,d)
        cls = self.cls_token.expand(B,1,-1)
        x   = torch.cat([cls, tok], dim=1) + self.pos_embed

        attn_maps = []
        for layer in self.layers:
            if self.return_attn:
                x, w = layer(x, need_weights=True)   # (B,22,22) per head
                attn_maps.append(w)                  # list[n_layers]  (B, n_heads, 22, 22)
            else:
                x = layer(x)

        if self.return_attn:
            return x[:,0], attn_maps                 # CLS, 전체 맵
        return x[:,0]

class GestureModel(nn.Module):
    def __init__(self, d_model=128, lstm_h=256, n_cls=8, return_attn=False):    # 8 Gestures
        super().__init__()
        self.return_attn = return_attn
        self.frame_enc = FrameEncoder(d_model, return_attn=return_attn)
        self.lstm = nn.LSTM(d_model, lstm_h, num_layers=2, batch_first=True)
        self.fc   = nn.Linear(lstm_h, n_cls)

    def forward(self, frames):                  # (B,30,21,3)
        B,T = frames.shape[:2]
        x = frames.view(B*T,21,3)
        if self.return_attn:
            feat, attn = self.frame_enc(x)      # feat:(B*T,d)
        else:
            feat = self.frame_enc(x)
        feat = feat.view(B,T,-1)
        out,_ = self.lstm(feat)
        logits = self.fc(out[:,-1])

        if self.return_attn:
            # attn : [n_layers][B*T, n_heads, 22, 22] → 원하는 대로 reshape
            return logits, attn
        return logits
