from simulation import SimState


class SimulationView:
    def __init__(self):
        self.simulation = SimState()

        self.prey = []
        self.predator = []

        self.linear = []
        self.K = 0

        self.collect()

    def supply(self, food_value, water_value):
        self.simulation.food = food_value
        self.simulation.water = water_value

    def collect(self):
        (prey_i, predator_i) = self.simulation.get()
        self.prey.    append(prey_i)
        self.predator.append(predator_i)
        self.linear  .append(self.K)
        self.K += 1

    def step(self, food_value, water_value):
        self.supply(food_value, water_value)
        self.simulation.step()
        self.collect()
