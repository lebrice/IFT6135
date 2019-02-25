"""
IFT6135 - Winter 2019
Arnold Kokoroko
Fabrice Normandin
Jérome Parent-Lévesque
"""

import torch
import torch.nn as nn

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
