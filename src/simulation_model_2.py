import math

from stable_baselines3 import PPO


class SimulationModel2:
    def __init__(self):
        self.model = PPO.load("distillation/model.zip")

    def act(self, x):
        x = [((math.log(x_i, 10) - 1) / 6 if x_i > 0 else 0) for x_i in x]
        y, _ = self.model.predict(x)
        food = y[0] * 1e5
        return [0 if food <= 10 else (math.log(food, 10) - 1.0) / 6]
