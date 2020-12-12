import math

from stable_baselines3 import PPO


def load_trained_model():
    model = PPO.load("logs/best_model.zip")

    def func(x):
        num_prey = int(math.pow(10, x[0] * 6 + 1))
        num_pred = int(math.pow(10, x[1] * 6 + 1))

        food = model.predict([num_prey, num_pred], deterministic=True)
        food = food[0] * 1e5

        if food <= 10:
            food = 0
        else:
            food = (math.log(food, 10) - 1.0) / 6
        return food

    return func
