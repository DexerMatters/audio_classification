import torch.nn as nn
import torch


class AudioTransformer(nn.Module):
    def __init__(
        self,
        num_mel_bins=128,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        max_seq_len=1000,
    ):
        super().__init__()
        self.input_projection = nn.Linear(num_mel_bins, hidden_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len + 1, hidden_size))
        self.max_seq_len = max_seq_len

        # Add input dropout
        self.input_dropout = nn.Dropout(dropout)

        # L2 normalization for better regularization
        self.layer_norm = nn.LayerNorm(hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 2,  # Reduced from 4x
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Add dropout before classification
        self.final_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Truncate sequence if it exceeds max_seq_len
        if seq_len > self.max_seq_len:
            x = x[:, : self.max_seq_len, :]
            seq_len = self.max_seq_len

        x = self.input_projection(x)  # Project to hidden size
        x = self.input_dropout(x)  # Apply dropout

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # Add CLS token
        x = x + self.pos_embedding[:, : seq_len + 1]  # Add positional embeddings

        x = self.layer_norm(x)  # Apply layer normalization
        x = self.transformer(x)  # Transformer processing

        cls_output = x[:, 0, :]  # Extract CLS token output
        cls_output = self.final_dropout(cls_output)  # Apply final dropout
        logits = self.classifier(cls_output)  # Classification
        return logits
