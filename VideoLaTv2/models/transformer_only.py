import torch
import torch.nn as nn
import math

class LandmarkTransformer(nn.Module):
    def __init__(
        self, num_joints=21, d_model=64, n_heads=4, n_layers=2, 
        max_frames=30, dropout_rate=0.3, num_classes=8, num_angles=15
    ):
        super().__init__()
        self.num_joints = num_joints
        self.num_angles = num_angles

        self.joint_pos_embed = nn.Parameter(torch.zeros(1, num_joints, 4))

        self.input_size = num_joints * 4 + num_angles
        self.input_proj = nn.Linear(self.input_size, d_model)

        pe = self._get_sinusoid_encoding(max_frames, d_model)  
        self.register_buffer('pos_embed', pe.unsqueeze(0)) 

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4*d_model,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.temp_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x): 
        B, T, D = x.shape
        coords = x[:, :, : self.num_joints * 4]
        angles = x[:, :, self.num_joints * 4 :]
        coords = coords.view(B, T, self.num_joints, 4)
        coords = coords + self.joint_pos_embed   
        coords = coords.view(B, T, self.num_joints * 4)
        x = torch.cat([coords, angles], dim=-1)  

        x = self.input_proj(x)            
        x = x + self.pos_embed[:, :T, :]

        x = self.transformer(x)            
        x = self.temp_norm(x)
        x = self.dropout(x)

        feat = x[:, -1, :]            # (B, d_model)
        logits = self.fc(feat)        # (B, num_classes)
        return logits

    @staticmethod
    def _get_sinusoid_encoding(n_pos, d_model):
        """
        Generate sinusoidal positional encodings 
            as in "Attention Is All You Need".
        Returns a tensor of shape (n_pos, d_model).
        """
        pe = torch.zeros(n_pos, d_model)
        position = torch.arange(n_pos, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() 
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
