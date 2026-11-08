import os
import mlx.core as mx
import numpy as np


def on_layer(self, i, m_in, act):
    H, W, G = self.H, self.W, self.G
    x = m_in.reshape(-1, H).astype(mx.float32)
    a = mx.abs(act.reshape(-1, W).astype(mx.float32))
    s = self.acc[i]
    s["xtx"] = s["xtx"] + x.T @ x                       # ridge router normal equations
    s["xtg"] = s["xtg"] + x.T @ a.reshape(-1, W // G, G).sum(-1)  # per-group activation targets
    s["neuron"] = s["neuron"] + a.sum(0)                # total |act| per neuron
    s["sample"].append(act.reshape(-1, W)[::8].astype(mx.float16))  # every 8th token, fp16 to save RAM
    mx.eval(s["xtx"], s["xtg"], s["neuron"], s["sample"][-1])  # materialize: keep graph shallow
