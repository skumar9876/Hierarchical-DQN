import numpy as np
import random

class QLearningAgent(object):

    DISCOUNT = 0.95

    def __init__(self, num_states, num_actions, learning_rate, epsilon=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.q_table = np.zeros((self.num_states, self.num_actions))
        self.curr_transition = None
        self.epsilon_decay_steps = 5000
        self.epsilons = np.linspace(1.0, 0.01, self.epsilon_decay_steps)
        self.curr_time_step = 0

        # self.learning_rate = 0.5

    def get_avaiable_actions(self, state):
        state_index = np.where(np.squeeze(state) == 1)[0][0]
        available_actions = []
        for i in xrange(len(state)):
            if i != state_index:
                available_actions.append(i)

        return available_actions

    def compute_state_index(self, state):
        if np.sum(state) == 1:
            state_index = np.where(np.squeeze(state) == 1)[0][0]
            return state_index
        else:
            # State vector contains an extra bit at the end.
            state_index = np.where(np.squeeze(state) == 1)[0][0]
            return self.num_actions + state_index

    def sample(self, state):
        state_index = self.compute_state_index(state)
        q_values = self.q_table[state_index]
        self.curr_time_step += 1

        # print state
        # print np.squeeze(state)
        # print state_index
        # print q_values
        # print np.argmax(q_values)
        # print ""
        # available_actions = self.get_avaiable_actions(state)
        e = self.epsilons[min(self.curr_time_step, self.epsilon_decay_steps - 1)]
        e = random.random()
        if e < self.epsilon:
            # return available_actions[random.randint(0, self.num_actions - 1)]
            return random.randint(0, self.num_actions - 1)
        else:
            # return available_actions[np.argmax(q_values)]
            return np.argmax(q_values)

    def best_action(self, state):
        # available_actions = self.get_avaiable_actions(state)

        state_index = self.compute_state_index(state)
        q_values = self.q_table[state_index]

        # return available_actions[np.argmax(q_values)]
        return np.argmax(q_values)

    def store(self, state, action, reward, next_state, terminal, eval, curr_reward):
        if not eval:
            self.curr_transition = [state, action, reward, next_state, terminal]
            self.curr_reward = curr_reward

    def update(self):
        state = self.curr_transition[0]
        action = self.curr_transition[1]
        reward = self.curr_transition[2]
        next_state = self.curr_transition[3]
        terminal = self.curr_transition[4]

        state_index = self.compute_state_index(state)
        next_state_index = self.compute_state_index(next_state)

        td_target = reward + (1 - terminal) * self.DISCOUNT * np.max(self.q_table[next_state_index])

        if self.curr_reward >= 1:
            print "Updating!"
            print state
            print action
            print reward
            print next_state
            print terminal
            print td_target
            print ""

        self.q_table[state_index, action] = (
            1 - self.learning_rate) * self.q_table[state_index, action] + self.learning_rate * td_target
