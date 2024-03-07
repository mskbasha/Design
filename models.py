import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


class DetectSpan(nn.Module):
    """Model to detect span in a video

    Args:
        input_shape (int): input shape to the model
    """

    def __init__(self, input_shape):
        """Init method for DetectSpan."""
        super().__init__()
        self.encoder = BertModel(BertConfig()).encoder
        self.project = nn.Linear(input_shape, 768)
        self.classification_layer = nn.Linear(768, 3)
        self.sigmoid = nn.Sigmoid()
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x, labels=None):
        x = self.project(x)
        x = self.encoder(x)
        x = self.classification_layer(x)
        x = self.sigmoid(x)
        if labels:
            return self.loss(x, labels)
        return x

    def loss(self, model_outputs, labels):
        self.loss_function(model_outputs, labels)
