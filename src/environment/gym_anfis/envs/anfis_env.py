import gym
import numpy as np

from gym_anfis.simulation_view import SimulationView


def get_simulation_view():
    return SimulationView([8000, 2000], [0, 0])


class Environment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(Environment, self).__init__()
        self.reward_range = (0, 1)
        self.action_space = gym.spaces.Box(np.array([0]), np.array([1]), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1e5, shape=(2,))
        self.simulation_view = get_simulation_view()
        self.simulation_step = 0

    def step(self, action):
        self.last_action = action
        old_obs = list(self.simulation_view.simulation.entityCount)
        self.simulation_view.step([action[0] * 1e5, 0])
        obs = np.array(self.simulation_view.simulation.entityCount)
        # reward = 1e5 - np.abs(self.simulation_view.simulation.entityCount[0] - self.simulation_view.simulation.entityCount[1])
        # reward = self.simulation_step
        # reward = 1
        # reward = 1 - min(abs(1 - np.average(np.array(obs) / np.array(old_obs))), 1)
        reward = 1 if action[0] > 0.4 and action[0] < 0.6 else 0
        # reward = -1 if np.any(np.array(self.simulation_view.simulation.entityCount) < 100) else 1
        done = bool(np.any(np.array(self.simulation_view.simulation.entityCount) < 100))
        if self.simulation_step > 100:
            done = True
        self.simulation_step += 1

        return obs, reward, done, {}

    def reset(self):
        self.simulation_view = get_simulation_view()
        self.simulation_step = 0
        return np.array(self.simulation_view.simulation.entityCount)

    def render(self, mode='human', close=False):
        print([list(self.simulation_view.simulation.entityCount), self.last_action])
