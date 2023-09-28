import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.max_size = max_size
        self.counter = 0
        self.obs_mem = np.zeros((self.max_size, input_shape))
        self.new_obs_mem = np.zeros((self.max_size, input_shape))
        self.action_mem = np.zeros((self.max_size, n_actions))
        self.reward_mem = np.zeros(self.max_size)
        self.done_mem = np.zeros(self.max_size, dtype=bool)

    def store(self, observation, action, reward, new_observation, done):
        index = self.counter % self.max_size
        self.counter += 1

        self.obs_mem[index] = observation
        self.new_obs_mem[index] = new_observation
        self.action_mem[index] = action
        self.reward_mem[index] = reward
        self.done_mem[index] = done

    def sample_buffer(self, batch_size):
        max_index = min(self.counter, self.max_size)

        sample = np.random.choice(max_index, batch_size)

        observations = self.obs_mem[sample]
        new_observations = self.new_obs_mem[sample]
        actions = self.action_mem[sample]
        rewards = self.reward_mem[sample]
        done_flags = self.done_mem[sample]

        return observations, actions, rewards, new_observations, done_flags
