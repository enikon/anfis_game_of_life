import membership
import anfis
import torch
import numpy as np


def createModel():
    invardefs = [
        ('prey_count', membership.make_bell_mfs(3.33333, 1.1, [-10, -3.333333, 3.333333, 10])),
        ('predator_count', membership.make_bell_mfs(3.33333, 1.1, [-10, -3.333333, 3.333333, 10])),
    ]
    outvars = ['food']
    model =  anfis.AnfisNet('Simple model', invardefs, outvars)
    coeff = torch.tensor([
        [0.2167,   0.7233, -0.0365],
        [0.2141,   0.5704, -0.4826],
        [-0.0683,  0.0022,  0.6495],
        [-0.2616,  0.9190, -2.9931],
        [-0.3293, -0.8943,  1.4290],
        [2.5820,  -2.3109,  3.7925],
        [0.8797,  -0.9407,  2.2487],
        [-0.8417, -1.5394, -1.5329],
        [-0.6422, -0.4384,  0.9792],
        [1.5534,  -0.0542, -4.7256],
        [-0.6864, -2.2435,  0.1585],
        [-0.3190, -1.3160,  0.9689],
        [-0.3200, -0.4654,  0.4880],
        [4.0220,  -3.8886,  1.0547],
        [0.3338,  -0.3306, -0.5961],
        [-0.5572,  0.9190, -0.8745],
    ])
    model.coeff = coeff.unsqueeze(1)
    return model


def executeModel(model, x):
    x_tensor = torch.Tensor(np.array([x]))
    y = model(x_tensor)
    return y.item()
