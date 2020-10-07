import gym
import numpy as np

from simulation_view import SimulationView


class Environment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(Environment, self).__init__()
        self.reward_range = (0, 1)
        self.action_space = gym.spaces.Box(np.array([0]), np.array([1]), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1e5, shape=(2,))
        self.simulation_view = self.__get_simulation_view()
        self.simulation_step = 0

    def step(self, action):
        old_obs = list(self.simulation_view.simulation.entityCount)
        self.simulation_view.step([action[0] * 1e5, 0])
        obs = list(self.simulation_view.simulation.entityCount)
        reward = 1 - min(abs(1 - np.average(np.array(obs) / np.array(old_obs))), 1)
        done = np.any(np.array(self.simulation_view.simulation.entityCount) < 100)
        if self.simulation_step > 1000:
            done = True
        self.simulation_step += 1

        return obs, reward, done, {}

    def reset(self):
        self.simulation_view = self.__get_simulation_view()
        self.simulation_step = 0
        return list(self.simulation_view.simulation.entityCount)

    def render(self, mode='human', close=False):
        print(list(self.simulation_view.simulation.entityCount))

    @staticmethod
    def __get_simulation_view():
        return SimulationView([8000, 2000], [0, 0])
