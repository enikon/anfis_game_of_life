import numpy as np

from gym_anfis.simulation import SimState


class SimulationView:
    def __init__(self, entities=None, resources=None):
        [prey_num, predator_num] = entities
        [food_value, water_value] = resources

        self.simulation = SimState(
            [prey_num, predator_num],
            [food_value]
        )
        self.decision = [0., 0.]

        self.prey = np.empty(shape=0)
        self.predator = np.empty(shape=0)
        self.food = np.empty(shape=0)
        self.food_decision = np.empty(shape=0)

        self.linear = np.empty(shape=0)
        self.K = 0

    def get(self):
        return self.simulation.getEntities()

    def supply(self, resources):
        self.decision = resources.copy()
        food_value, water_value = resources

        # self.simulation.resourceLevels[0] += food_value
        self.simulation.resourceLevels = [food_value]

    def collect(self):
        [prey_i, predator_i] = self.simulation.getEntities()
        [food_i] = [self.simulation.getResources()]
        [food_decision_i, water_decision] = self.decision

        self.prey = np.append(self.prey, [prey_i])
        self.predator = np.append(self.predator, [predator_i])
        self.food = np.append(self.food, [food_i])
        self.food_decision = np.append(self.food_decision, [food_decision_i])
        self.linear = np.append(self.linear, [self.K])
        self.K += 1

    def step(self, resources):
        self.supply(resources)
        self.simulation.step()
        self.collect()

    def reset(self, entities, resources):
        self.__init__(entities, resources)
