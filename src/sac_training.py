from training import Training
from anfis_model import train_sac
import math
import numpy as np


class SACTraining(Training):

    def __init__(self, environment):
        super().__init__()
        self.environment = environment
        self.models = None
        self.parameters_sets_count = None
        self.parameters_sets_total_count = 0
        self.parameters_count = 0

    def train(self, simulation_model):
        self.models = simulation_model.models
        self.parameters_count = simulation_model.parameters_count
        self.parameters_sets_count = simulation_model.parameters_sets_count
        self.parameters_sets_total_count = simulation_model.parameters_sets_total_count

        train_sac(self.models,
                  epochs=10, max_steps=1000, simulation=self.environment,
                  learning_rate=1-1e-3)


def log_prob_loss(actions_prob, action):
    log_prob = - ((action - actions_prob[0]) ** 2) / (2*np.clamp(actions_prob[1], min=1e-3))
    entropy = - math.log(math.sqrt(2 * math.pi * actions_prob[1]))
    return log_prob + entropy
