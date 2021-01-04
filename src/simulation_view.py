from simulation import SimState
import numpy as np


class SimulationView:
    def __init__(self, entities=None, resources=None):
        [prey_num, predator_num, plant_num] = entities
        [water_value, _] = resources

        self.simulation = SimState(
            [prey_num, predator_num, plant_num],
            [water_value]
        )
        self.decision = [0., 0.]

        self.prey = np.empty(shape=0)
        self.predator = np.empty(shape=0)
        self.plant = np.empty(shape=0)
        self.water = np.empty(shape=0)
        self.water_decision = np.empty(shape=0)

        self.linear = np.empty(shape=0)
        self.K = 0

    def get(self):
        return self.simulation.getEntities()

    def supply(self, resources):
        self.decision = resources.copy()
        food_value, water_value = resources

        #self.simulation.resourceLevels[0] += food_value
        self.simulation.resourceLevels = [food_value]

    def collect(self):
        [prey_i, predator_i, plant_i] = self.simulation.getEntities()
        [water_i] = [self.simulation.getResources()]
        [water_decision_i, _] = self.decision

        self.prey     = np.append(self.prey, [prey_i])
        self.predator = np.append(self.predator, [predator_i])
        self.plant = np.append(self.plant, [plant_i])
        self.water = np.append(self.water, [water_i])
        self.water_decision = np.append(self.water_decision, [water_decision_i])
        self.linear   = np.append(self.linear, [self.K])
        self.K += 1

    def step(self, resources):
        self.supply(resources)
        self.simulation.step()
        self.collect()

    def reset(self, entities, resources):
        self.__init__(entities, resources)
