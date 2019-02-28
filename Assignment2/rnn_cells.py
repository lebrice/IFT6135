#!/usr/bin/python3
"""
IFT6135 - Winter 2019
Arnold Kokoroko
Fabrice Normandin
Jérome Parent-Lévesque


Pytorch implementation of a 'Vanilla' RNN Cell and of a GRU.
"""
import torch
import torch.nn as nn

from utils import glorot_init, Dense
from typing import Tuple, List


class BaseRNNCell(torch.nn.Module):
    """
    Base class for VanillaRNNCell and GRUCell.

    Both classes have the same parameters and method docstrings documentation.
    Having a base class keeps common things in only one place. 
    """
    def __init__(self, input_size: int, hidden_size: int, output_size=None, dropout_keep_prob=1.0):
        """
        Creates a Simple-RNN cell, including a dense-dropout layer for computing the logits.

        input_size:     the size of the input
        hidden_size:    the number of hidden units, (also the size of the hidden_state tensor)
        output_size:    the size of the output logit vector (when None, is set to be the same as 'hidden_size')
        dropout_keep_prob:  The keep_chance for the dropout layer in the logits branch.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = hidden_size if output_size is None else output_size
        self.dropout_keep_prob = dropout_keep_prob

        self._dense = Dense(
            input_size=self.input_size,
            hidden_units=self.output_size,
            dropout_keep_prob=self.dropout_keep_prob
        )


    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the logits and next state for a given batch of inputs and an initial state.

        Params:
            x (torch.Tensor): Tensor of size [batch_size, input_size]
            hidden_state (torch.Tensor): Tensor of size [1, hidden_size]. Is used as the initial state.

        Returns:
            - Logits (torch.Tensor): Tensor of shape [batch_size, output_size]. No activation is used.
            - state (torch.Tensor): Tensor of shape [1, hidden_size], the hidden state after the timestep.
        """
        raise NotImplementedError("Use either VanillaRNNCell or GRUCell")


class VanillaRNNCell(BaseRNNCell):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear_x = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.linear_h = nn.Linear(self.hidden_size, self.hidden_size)
    
    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self._dense(hidden_state)
        h = torch.tanh(self.linear_x(x) + self.linear_h(hidden_state))
        return y, h


class GRURNNCell(BaseRNNCell):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reset = nn.Linear(self.input_size, self.hidden_size)
        self._reset_u = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self._forget = nn.Linear(self.input_size, self.hidden_size)
        self._forget_u = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self._state = nn.Linear(self.input_size, self.hidden_size)
        self._state_u = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, x_t: torch.Tensor, h_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r_t = torch.sigmoid(self._reset(x_t) + self._reset_u(h_t))
        z_t = torch.sigmoid(self._forget(x_t) + self._forget_u(h_t))
        h_t_tilda = torch.tanh(self._state(x_t) + self._state_u(r_t * h_t))

        h_t = (1 - z_t) * h_t + z_t * h_t_tilda
        y = self._dense(h_t)
        return y, h_t
