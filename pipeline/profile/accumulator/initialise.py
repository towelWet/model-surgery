import os
import mlx.core as mx
import numpy as np


def initialise(self, n_layers, H, W, G):
    self.H, self.W, self.G = H, W, G                    # hidden, intermediate, quant group width
    self.acc = {i: dict(xtx=mx.zeros((H, H)), xtg=mx.zeros((H, W // G)), neuron=mx.zeros((W,)), sample=[])
                for i in range(n_layers)}
