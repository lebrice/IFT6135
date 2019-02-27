#!/usr/bin/python3
"""
IFT6135 - Winter 2019
Arnold Kokoroko
Fabrice Normandin
Jérome Parent-Lévesque


GRU RNN implementation in Pytorch.
"""
import torch
import torch.nn as nn
from typing import Tuple
from utils import one_hot_encoding, glorot_init


class GRU(nn.Module):  # Implement a stacked GRU RNN
    """
    Follow the same instructions as for RNN (above), but use the equations for 
    GRU, not Vanilla RNN.
    """

    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
        super(GRU, self).__init__()

        # TODO ========================

    def init_weights_uniform(self):
        # TODO ========================
        pass

    def init_hidden(self):
        # TODO ========================
        # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)
        return None

    def forward(self, inputs, hidden):
        # TODO ========================
        return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden

    def generate(self, input, hidden, generated_seq_len):
        # TODO ========================
        return samples


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        output_size = input_size

        self._reset = nn.Linear(input_size, hidden_size)
        self._reset_u = nn.Linear(hidden_size, hidden_size, bias=False)

        self._forget = nn.Linear(input_size, hidden_size)
        self._forget_u = nn.Linear(hidden_size, hidden_size, bias=False)

        self._state = nn.Linear(input_size, hidden_size)
        self._state_u = nn.Linear(hidden_size, hidden_size, bias=False)

        self._output = nn.Linear(hidden_size, output_size)

        self.hidden_state = nn.Parameter(torch.Tensor(1, hidden_size))

        self.init_weights()

    def init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                glorot_init(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the forward-pass, updating the hidden_state after each slice.

        Params:
            x (torch.Tensor): Tensor of size [batch_size, sequence_length, vocab_size]
            hidden_state (torch.Tensor, optional): Tensor of size [TODO]. If provided, is used as the initial state.
            When omitted, the model's internal hidden_state tensor is used instead. 

        Returns:
            A Tuple of:
                - Y (torch.Tensor): the output sequence, same size as x.
                - h_t (torch.Tensor): the final state after evaluating the sequence.
        """
        h_t = self.hidden_state if hidden_state is None else hidden_state
        
        x = x.transpose(0, 1)  # Flip the tensor to [seq, batch, vocab_size].
        y = torch.Tensor(*x.size())  # will hold the outputs

        for t, x_t in enumerate(x):
            # calculate output at time t.
            r_t = torch.sigmoid(self._reset(x_t) + self._reset_u(h_t))
            z_t = torch.sigmoid(self._forget(x_t) + self._forget_u(h_t))
            h_t_tilda = torch.tanh(self._state(x_t) + self._state_u(r_t * h_t))

            h_t = (1 - z_t) * h_t + z_t * h_t_tilda
            y[t] = torch.softmax(self._output(h_t), dim=1)

        y = y.transpose(0, 1)  # switch back to [batch, seq, vocab_size].

        # TODO: Should we update the hidden_state attribute? If so, should we
        # do it after each batch? or after each example ? Do we do also do it
        # if an initial hidden_state was passed as an argument? Should we
        # overwrite the attribute with the passed in parameter?
        if hidden_state is None:
            self.hidden_state.data = h_t
        return y, h_t