import numpy as np
import random


class ReplayBuffer(object):

    def __init__(self, max_size, init_size, batch_size):
        self.max_size = max_size
        self.init_size = init_size
        self.batch_size = batch_size

        self.states = np.array([None] * self.max_size)
        self.actions = np.array([None] * self.max_size)
        self.rewards = np.array([None] * self.max_size)
        self.next_states = np.array([None] * self.max_size)
        self.terminals = np.array([None] * self.max_size)

        self.curr_pointer = 0
        self.curr_size = 0

    def add(self, state, action, reward, next_state, terminal):
        self.states[self.curr_pointer] = np.squeeze(state)
        self.actions[self.curr_pointer] = action
        self.rewards[self.curr_pointer] = reward
        self.next_states[self.curr_pointer] = np.squeeze(next_state)
        self.terminals[self.curr_pointer] = terminal

        self.curr_pointer += 1
        self.curr_size = min(self.max_size, self.curr_size + 1)
        # If replay buffer is full, set current pointer to be at the beginning of the buffer.
        if self.curr_pointer >= self.max_size:
            self.curr_pointer -= self.max_size

    def sample(self):
        if self.curr_size < self.init_size:
            return [], [], [], [], []
        sample_indices = []

        # Ensure that the most recent transition is in the returned batch.
        sample_indices.append(self.curr_pointer - 1)
        for i in xrange(self.batch_size - 1):
            sample_indices.append(random.randint(0, self.curr_size - 1))

        returned_states = []
        returned_actions = []
        returned_rewards = []
        returned_next_states = []
        returned_terminals = []

        for i in xrange(len(sample_indices)):
            index = sample_indices[i]
            returned_states.append(self.states[index])
            returned_actions.append(self.actions[index])
            returned_rewards.append(self.rewards[index])
            returned_next_states.append(self.next_states[index])
            returned_terminals.append(self.terminals[index])

        return np.array(returned_states), np.array(returned_actions), np.array(
            returned_rewards), np.array(returned_next_states), np.array(returned_terminals)
        # return self.states[sample_indices], self.actions[sample_indices], self.rewards[sample_indices], self.next_states[sample_indices], self.terminals[sample_indices]