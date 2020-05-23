import torch
from torch.utils.data import TensorDataset, DataLoader
import simulation_model
import numpy as np


def train(model, data, epochs=10):
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.99)
    criterion = torch.nn.MSELoss(reduction='sum')
    for t in range(epochs):
        for x, y_actual in data:
            y_pred = model(x)
            loss = criterion(y_pred, y_actual)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        x, y_actual = data.dataset.tensors
        with torch.no_grad():
            model.fit_coeff(x, y_actual)

        if t % 1 == 0:
            y_pred = model(x)
            loss = criterion(y_pred, y_actual)
            print('epoch %d: loss=%f' % (t, loss.item()))


def makeMockData(size=10000):
    x = torch.tensor(np.random.randint(100, 100000, size=2 * size).reshape(-1, 2).astype(np.single))
    y = torch.tensor(np.full((size, 1), 3.0).astype(np.single))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=100, shuffle=True)


model = simulation_model.createModel()
data = makeMockData()
train(model, data)
simulation_model.saveModel(model)
