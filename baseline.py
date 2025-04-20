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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,  # Set batch_first to True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Truncate sequence if it exceeds max_seq_len
        if seq_len > self.max_seq_len:
            x = x[:, : self.max_seq_len, :]
            seq_len = self.max_seq_len

        x = self.input_projection(x)  # Project to hidden size
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # Add CLS token
        x = x + self.pos_embedding[:, : seq_len + 1]  # Add positional embeddings
        x = self.transformer(x)  # Transformer processing
        cls_output = x[:, 0, :]  # Extract CLS token output
        logits = self.classifier(cls_output)  # Classification
        return logits
