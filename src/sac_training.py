from training import Training
from anfis_model import train_sac


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
