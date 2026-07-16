import torch
import torch.nn as nn
from config import DEVICE, VOCABULARY_CAPACITY

class PolymorphicTransformerEngine(nn.Module):
    """Universal Transformer network capable of mapping variable sequence arrays into single outcomes."""
    def __init__(self, d_model: int = 32, nhead: int = 4, num_layers: int = 2):
        super(PolymorphicTransformerEngine, self).__init__()
        # Buffer space allocation window covering vocabulary map scales
        self.embedding = nn.Embedding(num_embeddings=VOCABULARY_CAPACITY + 100000, embedding_dim=d_model, padding_idx=0)
        self.pos_encoder = nn.Parameter(torch.randn(1, 200, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        embeddings = self.embedding(x) + self.pos_encoder[:, :seq_len, :]
        
        # Multi-Head Attention computation pass across the feature space
        transformed = self.transformer(embeddings, src_key_padding_mask=key_padding_mask)
        
        # Map values to conditional multiplier weightings
        mask_weights = (~key_padding_mask).unsqueeze(-1).float()
        pooled = torch.sum(transformed * mask_weights, dim=1) / torch.clamp(torch.sum(mask_weights, dim=1), min=1.0)
        
        return self.sigmoid(self.fc(pooled))
