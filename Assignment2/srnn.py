#!/usr/bin/python3
"""
IFT6135 - Winter 2019
Arnold Kokoroko
Fabrice Normandin
Jérome Parent-Lévesque


"Vanilla" RNN implementation in Pytorch.
"""
import torch
import torch.nn as nn

from utils import glorot_init

class SRNN(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        output_size = input_size  # TODO: output is same size as input, right?

        self.linear_x = nn.Linear(input_size, hidden_size, bias=False)
        self.linear_h = nn.Linear(hidden_size, hidden_size)
        self.linear_y = nn.Linear(hidden_size, output_size)

        self.hidden_state = nn.Parameter(torch.Tensor(1, hidden_size))

        self.init_weights()

    def init_weights(self):
        """
        Initialize all weights of the RNN.
        (Uses Glorot initializer for the kernels, and constant (0) for the biases.)
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                glorot_init(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # TODO: How should we init the hidden state ?
        glorot_init(self.hidden_state)

    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor = None):
        """
        Computes the forward-pass, updating the hidden_state after each slice.

        Params:
            x (torch.Tensor): Tensor of size [batch_size, sequence_length, vocab_size]
            hidden_state (torch.Tensor, optional): Tensor of size [? ? ?]. If provided, is used as the initial state.
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
            y[t] = torch.softmax(self.linear_y(self.hidden_state), dim=1)
            h_t = torch.tanh(self.linear_x(x_t) + self.linear_h(h_t))

        y = y.transpose(0, 1)  # switch back to [batch, seq, vocab_size].

        # TODO: Should we update the hidden_state attribute? If so, should we
        # do it after each batch? or after each example ? Do we do also do it
        # if an initial hidden_state was passed as an argument? Should we
        # overwrite the attribute with the passed in parameter?
        if hidden_state is None:
            self.hidden_state.data = h_t
        return y, h_t

