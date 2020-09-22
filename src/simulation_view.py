from simulation import SimState
import numpy as np
import math


class SimulationView:
    def __init__(self, entities=None, resources=None):
        self.simulation = SimState(
            entities,
            resources
        )
        self.decision = [0.]

        self.prey = np.empty(shape=0)
        self.predator = np.empty(shape=0)
        self.food = np.empty(shape=0)
        self.food_decision = np.empty(shape=0)

        self.linear = np.empty(shape=0)
        self.K = 0

        self._collect()

    def get(self):
        return [self.simulation.getEntities(), self.simulation.getResources()]

    def get_normalised(self):
        return [
            [self.__normalisation_function(e) for e in self.simulation.getEntities()],
            [self.__normalisation_function(r) for r in self.simulation.getResources()]
        ]

    def _supply(self, resources):
        self.decision = resources.copy()

        # self.simulation.resourceLevels[0] += food_value
        self.simulation.resourceLevels = resources.copy()

    def _collect(self):
        [prey_i, predator_i] = self.simulation.getEntities()
        [food_i] = [self.simulation.getResources()]

        self.prey = np.append(self.prey, [prey_i])
        self.predator = np.append(self.predator, [predator_i])
        self.food = np.append(self.food, [food_i])
        self.food_decision = np.append(self.food_decision, self.decision.copy())
        self.linear = np.append(self.linear, [self.K])
        self.K += 1

    def step(self, resources):
        self._supply(resources)
        reward, done = self.simulation.step()
        self._collect()
        return reward, done

    def step_normalised(self, resources):
        return self.step([self.__nominalisation_function(r) for r in resources])

    def reset(self, entities=None, resources=None):
        self.__init__(entities, resources)

    @staticmethod
    def __normalisation_function(x):
        t = max(10.0, min(1000000.0, x))
        return (math.log10(t) - 1) / 6

    @staticmethod
    def __nominalisation_function(x):
        t = max(0.0, min(1.0, x))
        return 10 ** (t * 6 + 1)

    def nominalise(self, x):
        return self.__nominalisation_function(x)
