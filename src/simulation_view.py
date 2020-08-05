from simulation import SimState
import numpy as np


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

        self._collect()

    def get(self):
        return [self.simulation.getEntities(), self.simulation.getResources()]

    def _supply(self, resources):
        self.decision = resources.copy()
        food_value, water_value = resources

        #self.simulation.resourceLevels[0] += food_value
        self.simulation.resourceLevels = [food_value]

    def _collect(self):
        [prey_i, predator_i] = self.simulation.getEntities()
        [food_i] = [self.simulation.getResources()]
        [food_decision_i, water_decision] = self.decision

        self.prey     = np.append(self.prey, [prey_i])
        self.predator = np.append(self.predator, [predator_i])
        self.food = np.append(self.food, [food_i])
        self.food_decision = np.append(self.food_decision, [food_decision_i])
        self.linear   = np.append(self.linear, [self.K])
        self.K += 1

    def step(self, resources):
        self._supply(resources)
        reward, done = self.simulation.step()
        self._collect()

        return reward, done

    def reset(self, entities=None, resources=None):
        self.__init__(entities, resources)
