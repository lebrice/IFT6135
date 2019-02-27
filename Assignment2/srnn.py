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
from typing import Tuple, List
from itertools import zip_longest


class SRNNCell(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size=None, dropout_keep_prob=1.0):
        """
        Creates a single Simple-RNN cell, including a dense-dropout layer for computing the logits.

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

        self.linear_x = nn.Linear(input_size, hidden_size, bias=False)
        self.linear_h = nn.Linear(hidden_size, hidden_size)

        self.dense_y = Dense(input_size, self.output_size,
                             dropout_keep_prob=dropout_keep_prob)

    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor):
        """
        Computes the logits and next state for a given batch of inputs and an initial state.

        Params:
            x (torch.Tensor): Tensor of size [batch_size, vocab_size]
            hidden_state (torch.Tensor): Tensor of size [1, hidden_size]. Is used as the initial state.

        Returns:
            - Logits (torch.Tensor): Tensor of shape [batch_size, output_size]. No activation is used.
            - state (torch.Tensor): Tensor of shape [1, hidden_size], the hidden state after the timestep.
        """
        y = self.dense_y(hidden_state)
        h = torch.tanh(self.linear_x(x) + self.linear_h(hidden_state))
        return y, h


class Dense(nn.Module):
    def __init__(self, input_size: int, hidden_units: int, dropout_keep_prob=1.0):
        self.linear = nn.Linear(input_size, hidden_units)
        self.dropout = nn.Dropout(1 - dropout_keep_prob)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear(inputs))


# Implement a stacked vanilla RNN with Tanh nonlinearities.
class RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
        """
        emb_size:     The number of units in the input embeddings
        hidden_size:  The number of hidden units per layer
        seq_len:      The length of the input sequences
        vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
        num_layers:   The depth of the stack (i.e. the number of hidden layers at 
                      each time-step)
        dp_keep_prob: The probability of *not* dropping out units in the 
                      non-recurrent connections.
                      Do not apply dropout on recurrent connections.
        """

        # TODO ========================
        # Initialization of the parameters of the recurrent and fc layers.
        # Your implementation should support any number of stacked hidden layers
        # (specified by num_layers), use an input embedding layer, and include fully
        # connected layers with dropout after each recurrent layer.
        # Note: you may use pytorch's nn.Linear, nn.Dropout, and nn.Embedding
        # modules, but not recurrent modules.
        #
        # To create a variable number of parameter tensors and/or nn.Modules
        # (for the stacked hidden layer), you may need to use nn.ModuleList or the
        # provided clones function (as opposed to a regular python list), in order
        # for Pytorch to recognize these parameters as belonging to this nn.Module
        # and compute their gradients automatically. You're not obligated to use the
        # provided clones function.

        super().__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dp_keep_prob = dp_keep_prob

        self.embedding_layer = nn.Embedding(self.emb_size, self.vocab_size)

        self.recurrent_layers: List[SRNNCell] = nn.ModuleList([
            SRNNCell(
                input_size=self.emb_size if i == 0 else self.hidden_size,
                hidden_size=self.hidden_size,
                output_size=self.hidden_size if i < self.num_layers - 1 else self.vocab_size,
                dropout_keep_prob=self.dp_keep_prob if i < self.num_layers - 1 else 1.0
            ) for i in range(self.num_layers)
        ])

    def init_weights_uniform(self):
        # TODO ========================
        # Initialize all the weights uniformly in the range [-0.1, 0.1]
        # and all the biases to 0 (in place)
        for module in self.modules():
            if hasattr(module, "weight") and module.weight is not None:
                # TODO: weren't we instructed to use Glorot init in the assignment instructions?
                nn.init.uniform_(module.weight, -0.1, 0.1)
                # glorot_init(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def init_hidden(self) -> torch.Tensor:
        """
        This is used for the first mini-batch in an epoch, only.
        """
        # TODO ========================
        # initialize the hidden states to zero
        # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            - inputs: A mini-batch of input sequences, composed of integers that 
                        represent the index of the current token(s) in the vocabulary.
                            shape: (seq_len, batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)

        Returns:
            - Logits for the softmax over output tokens at every time-step.
                    **Do NOT apply softmax to the outputs!**
                    Pytorch's CrossEntropyLoss function (applied in ptb-lm.py) does 
                    this computation implicitly.
                        shape: (seq_len, batch_size, vocab_size)
            - The final hidden states for every layer of the stacked RNN.
                    These will be used as the initial hidden states for all the 
                    mini-batches in an epoch, except for the first, where the return 
                    value of self.init_hidden will be used.
                    See the repackage_hiddens function in ptb-lm.py for more details, 
                    if you are curious.
                        shape: (num_layers, batch_size, hidden_size)
        """
        # TODO ========================
        # Compute the forward pass, using a nested python for loops.
        # The outer for loop should iterate over timesteps, and the
        # inner for loop should iterate over hidden layers of the stack.
        #
        # Within these for loops, use the parameter tensors and/or nn.modules you
        # created in __init__ to compute the recurrent updates according to the
        # equations provided in the .tex of the assignment.
        #
        # Note that those equations are for a single hidden-layer RNN, not a stacked
        # RNN. For a stacked RNN, the hidden states of the l-th layer are used as
        # inputs to to the {l+1}-st layer (taking the place of the input sequence).

        # Tensor to hold the outputs.
        logits = torch.Tensor(self.seq_len, self.batch_size, self.vocab_size)

        for t, x_t in enumerate(inputs):
            # The Input of the first layer is the word batch at time t.
            x = x_t
            for layer, rnn_cell in enumerate(self.recurrent_layers):
                # compute the new outputs and state
                x, hidden[layer] = rnn_cell(x, hidden[layer])
            # the output of the last layer is kept.
            logits[t] = x
        return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden

    def generate(self, input, hidden, generated_seq_len):
        # TODO ========================
        # Compute the forward pass, as in the self.forward method (above).
        # You'll probably want to copy substantial portions of that code here.
        #
        # We "seed" the generation by providing the first inputs.
        # Subsequent inputs are generated by sampling from the output distribution,
        # as described in the tex (Problem 5.3)
        # Unlike for self.forward, you WILL need to apply the softmax activation
        # function here in order to compute the parameters of the categorical
        # distributions to be sampled from at each time-step.
        """
        Arguments:
            - input: A mini-batch of input tokens (NOT sequences!)
                            shape: (batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)
            - generated_seq_len: The length of the sequence to generate.
                            Note that this can be different than the length used 
                            for training (self.seq_len)
        Returns:
            - Sampled sequences of tokens
                        shape: (generated_seq_len, batch_size)
        """
        # Tensor to hold the outputs.
        samples = torch.Tensor(self.generated_seq_len, self.batch_size)
        x = input
        for t in range(generated_seq_len):
            # The Input of the first layer is the word batch at time t. 
            for layer, rnn_cell in enumerate(self.recurrent_layers):
                # compute the new outputs and state
                x, hidden[layer] = rnn_cell(x, hidden[layer])    
            # the output of the last layer is kept.
            prob = torch.softmax(x, dim=1)
            print(prob)
            values, indices = torch.max(prob, dim=1)
            samples[t] = indices
        return samples
