import math

from stable_baselines3 import PPO


def load_trained_model():
    model = PPO.load("logs/best_model.zip")

    def func(x):
        num_prey = int(math.pow(10, x[0] * 6 + 1))
        num_pred = int(math.pow(10, x[1] * 6 + 1))
        num_plant = int(math.pow(10, x[2] * 6 + 1))

        water = model.predict([num_prey, num_pred, num_plant], deterministic=True)
        water = water[0] * 1e5

        if water <= 10:
            water = 0
        else:
            water = (math.log(water, 10) - 1.0) / 6
        return water

    return func
