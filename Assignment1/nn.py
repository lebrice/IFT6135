#!/usr/bin/python3

"""
IFT 6135 - Representation Learning Assignment #1

Fabrice Normandin
Jerome Parent-Lévesque
Arnold Kokoroko
"""

from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)


class NN(object):

    def __init__(self, hidden_dims=(1024, 2048), n_hidden=2, mode='train', datapath=None, model_path=None):
        pass

    def initialize_weights(self, n_hidden, dims):
        pass

    def forward(self, input, labels, *args):
        pass

    def activation(self, input):
        pass

    def loss(self, prediction, *args):
        pass

    def softmax(self, input, *args):
        pass

    def backward(self, cache, labels, *args):
        pass

    def update(self, grads, *args):
        pass

    def train(self):
        pass

    def test(self):
        pass
