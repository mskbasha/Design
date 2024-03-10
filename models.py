import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

torch.manual_seed(10)


class DetectSpan(nn.Module):
    """Model to detect span in a video

    Args:
        input_shape (int): input shape to the model
    """

    def __init__(
        self,
        input_shape,
        nhead=12,
        d_model=768,
        num_blocks=12,
        max_seq_len=1024,
        dropout=0.2,
    ):
        """Init method for DetectSpan."""
        super().__init__()
        self.position_embeddings = nn.Embedding(max_seq_len, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_blocks,
        )
        self.project = nn.Linear(input_shape, d_model)
        self.classification_layer = nn.Linear(d_model, 3)
        self.sigmoid = nn.Sigmoid()
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x, attention_mask, labels=None):
        x = self.project(x)
        seq_length = x.size(1)
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0)
        posisional_encodings = self.position_embeddings(positions)
        x += posisional_encodings
        x = self.encoder(x, src_key_padding_mask=attention_mask.to(torch.bool))
        x = self.classification_layer(x)
        x = self.sigmoid(x)
        if labels:
            return self.loss(x, labels)
        return x

    def loss(self, model_outputs, labels):
        self.loss_function(model_outputs, labels)
