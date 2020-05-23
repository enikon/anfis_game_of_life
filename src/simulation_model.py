from membership import BellMembFunc, TriangularMembFunc
import anfis
import torch
import numpy as np


def createModel():
    invardefs = [
        ('prey_count', [TriangularMembFunc(0, 1000, 2000), TriangularMembFunc(1000, 10000, 100000)]),
        ('predator_count', [TriangularMembFunc(0, 1000, 2000), TriangularMembFunc(1000, 10000, 100000)]),
    ]
    outvars = ['food']
    model = anfis.AnfisNet('Simple model', invardefs, outvars)
    coeff = torch.tensor([
        [0.001, 0.0001, -0.0365],
        [0.001, 0.0001, -0.4826],
        [0.001, 0.0001,  0.6495],
        [0.001, 0.0001, -2.9931]
    ])
    model.coeff = coeff.unsqueeze(1)
    return model


def saveModel(model):
    torch.save(model, 'model')


def loadModel():
    model = torch.load('model')
    model.eval()
    return model


def executeModel(model, x):
    x_tensor = torch.Tensor(np.array([x]))
    y = model(x_tensor)
    return y.item()
