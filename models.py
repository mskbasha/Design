import torch
import torch.nn as nn


class DetectSpan(nn.Module):
    """
    Model to detect span in a text sequence using BERT embeddings and Transformer Encoder.

    Args:
        input_shape (int): input shape to the model
        nhead (int): the number of heads in the multiheadattention models
        d_model (int): the number of expected features in the input (default: 768)
        num_blocks (int): the number of blocks in the encoder (default: 12)
        max_seq_len (int): maximum sequence length for positional embeddings (default: 1024)
        dropout (float): dropout probability (default: 0.2)
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
        super(DetectSpan, self).__init__()
        # Positional Embeddings
        self.position_embeddings = nn.Embedding(max_seq_len, d_model)
        # Transformer Encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_blocks,
        )
        # Linear projection for input shape
        self.project = nn.Linear(input_shape, d_model)
        # Classification layer
        self.classification_layer = nn.Linear(
            d_model, 3
        )  # 3 classes for token classification
        # Loss function
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x, attention_mask, labels=None):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_shape)
            attention_mask (torch.Tensor): Attention mask for padding tokens
            labels (torch.Tensor): True labels for the tokens (optional)

        Returns:
            torch.Tensor or float: Predicted probabilities or loss if labels are provided
        """
        # Project inputs to d_model space
        x = self.project(x)
        seq_length = x.size(1)
        # Generate positional encodings
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0)
        positional_encodings = self.position_embeddings(positions)
        # Add positional encodings to input
        x += positional_encodings
        # Pass through Transformer Encoder
        x = self.encoder(x, src_key_padding_mask=attention_mask.to(torch.bool))
        # Pass through classification layer
        x = self.classification_layer(x)
        # If labels provided, calculate and return loss
        if labels is not None:
            return self.calculate_loss(x, labels)
        return x

    def calculate_loss(self, model_outputs, labels):
        """
        Calculate the loss.

        Args:
            model_outputs (torch.Tensor): Predicted logits from the model
            labels (torch.Tensor): True labels

        Returns:
            float: Calculated loss
        """
        return self.loss_function(model_outputs.transpose(1, 2), labels)


class VideoClassifier(nn.Module):
    """
    Model to detect span in a text sequence using BERT embeddings and Transformer Encoder.

    Args:
        input_shape (int): input shape to the model
        nhead (int): the number of heads in the multiheadattention models
        d_model (int): the number of expected features in the input (default: 768)
        num_blocks (int): the number of blocks in the encoder (default: 12)
        max_seq_len (int): maximum sequence length for positional embeddings (default: 1024)
        dropout (float): dropout probability (default: 0.2)
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
        super(VideoClassifier, self).__init__()
        # Positional Embeddings
        self.position_embeddings = nn.Embedding(max_seq_len + 1, d_model)
        # Transformer Encoder
        self.max_seq_len = max_seq_len
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_blocks,
        )
        # Linear projection for input shape
        self.project = nn.Linear(input_shape, d_model)
        # Classification layer
        self.classification_layer = nn.Linear(
            d_model, 5
        )  # 3 classes for token classification
        # Loss function
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x, attention_mask, labels=None):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_shape)
            attention_mask (torch.Tensor): Attention mask for padding tokens
            labels (torch.Tensor): True labels for the tokens (optional)

        Returns:
            torch.Tensor or float: Predicted probabilities or loss if labels are provided
        """
        # Project inputs to d_model space
        x = self.project(x)
        cls_token = self.position_embeddings[self.max_seq_len]
        x = torch.cat([cls_token, x])
        seq_length = x.size(1)
        # Generate positional encodings
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0)
        positional_encodings = self.position_embeddings(positions)
        # Add positional encodings to input
        x += positional_encodings
        # Pass through Transformer Encoder
        x = self.encoder(x, src_key_padding_mask=attention_mask.to(torch.bool))
        # Pass through classification layer
        x = self.classification_layer(x[:, 0])
        # If labels provided, calculate and return loss
        if labels is not None:
            return self.calculate_loss(x, labels)
        return x

    def calculate_loss(self, model_outputs, labels):
        """
        Calculate the loss.

        Args:
            model_outputs (torch.Tensor): Predicted logits from the model
            labels (torch.Tensor): True labels

        Returns:
            float: Calculated loss
        """
        return self.loss_function(model_outputs.transpose(1, 2), labels)
