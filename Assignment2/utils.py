"""
IFT6135 - Winter 2019
Arnold Kokoroko
Fabrice Normandin
Jérome Parent-Lévesque
"""

import torch
import torch.nn as nn


class Dense(nn.Module):
    """
    Simple Dense layer with Dropout and linear activation.
    """
    def __init__(self, input_size: int, hidden_units: int, dropout_keep_prob=1.0):
        self.linear = nn.Linear(input_size, hidden_units)
        self.dropout = nn.Dropout(1 - dropout_keep_prob)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear(inputs))

def glorot_init(weight: torch.Tensor) -> None:
    nn.init.xavier_normal_(weight)

def one_hot_encoding(x: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    Returns the one-hot encoded version of the inputs.

    Args:
        x (torch.Tensor): Integer Tensor of size [batch_size, seq_len]
        vocab_size (int): The size of the vocabulary.
    Returns:
        a one-hot tensor of size [batch_size, seq_len, vocab_size]
    """
    x = x.type(torch.long)
    batch_size, seq_len = x.size()
    output = torch.zeros([batch_size, seq_len, vocab_size])
    for i, sequence in enumerate(x):
        for j, token_index in enumerate(sequence):
            output[i][j][token_index] = 1
    return output


def num_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
