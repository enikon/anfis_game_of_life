from simulation import SimState
import numpy as np


class SimulationView:
    def __init__(self, entities=None):
        self.simulation = SimState(entities)

        self.prey = np.empty(shape=0)
        self.predator = np.empty(shape=0)

        self.linear = np.empty(shape=0)
        self.K = 0

        self.collect()

    def get(self):
        return self.simulation.get()

    def supply(self, food_value, water_value):
        #self.simulation.resourceLevels[0] += food_value
        self.simulation.resourceLevels = [food_value]

    def collect(self):
        [prey_i, predator_i] = self.simulation.get()
        self.prey     = np.append(self.prey, [prey_i])
        self.predator = np.append(self.predator, [predator_i])
        self.linear   = np.append(self.linear, [self.K])
        self.K += 1

    def step(self, food_value, water_value):
        self.supply(food_value, water_value)
        self.simulation.step()
        self.collect()

    def reset(self, entities):
        self.simulation = SimState(entities)
        self.prey = np.empty(shape=0)
        self.predator = np.empty(shape=0)
        self.linear = np.empty(shape=0)

        self.K = 0
        self.collect()
