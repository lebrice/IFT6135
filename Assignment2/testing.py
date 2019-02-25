#!/usr/bin/python3
"""
IFT6135 - Winter 2019
Arnold Kokoroko
Fabrice Normandin
Jérome Parent-Lévesque


"Vanilla" RNN implementation in Pytorch.
"""
import torch
import torch.nn

from srnn import SRNN
from gru import GRURNN

from utils import *


x1 = torch.Tensor([[0, 2, 3, 9]])
x2 = torch.Tensor([[0, 5, 7, 9]])

x_batch = torch.cat([x1, x2], 0)
x_batch = one_hot_encoding(x_batch, vocab_size=10)

def test(model):
    y, h = model(x_batch)
    print("num_params:", num_trainable_params(model))
    print("y:", y.size(), "h:", h.size())

model = torch.nn.RNN(input_size=10, hidden_size=10, batch_first=True)
test(model)
model = SRNN(input_size=10, hidden_size=10)
test(model)
model = GRURNN(input_size=10, hidden_size=6)
test(model)

# print("h:", h.size())
# print(y[0])
# print(y[0][0])
# print(y[0][1])
# print(y[0][2])
# print(y[0][3])
