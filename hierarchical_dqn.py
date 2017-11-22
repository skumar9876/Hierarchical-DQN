"""
Hierarchical DQN implementation as described in Kulkarni et al.
https://arxiv.org/pdf/1604.06057.pdf
@author: Saurabh Kumar
"""

from collections import defaultdict
from dqn import DqnAgent
import numpy as np
from qLearning import QLearningAgent
import sys


class HierarchicalDqnAgent(object):
    INTRINSIC_STEP_COST = -1    # Step cost for the controller.

    def __init__(self,
                 learning_rates=[0.1, 0.00025],
                 state_sizes=[0, 0],
                 subgoals=None,
                 num_subgoals=0,
                 num_primitive_actions=0,
                 meta_controller_state_fn=None,
                 check_subgoal_fn=None):

        """Initializes a hierarchical DQN agent.

           Args:
            learning_rates: learning rates of the meta-controller and controller agents.
            state_sizes: state sizes of the meta-controller and controller agents.
            subgoals: array of subgoals for the meta-controller.
            num_subgoals: the action space of the meta-controller.
            num_primitive_actions: the action space of the controller.
            meta_controller_state_fn: function that returns the state of the meta-controller.
            check_subgoal_fn: function that checks if agent has satisfied a particular subgoal.
        """

        # Note: States for meta-controller and controller are assumed to be 1-dimensional.
        self._meta_controller = DqnAgent(state_dims=state_sizes[0],
            num_actions=num_subgoals,
            learning_rate=learning_rates[0],
            epsilon_end=0.01)

        self._controller = DqnAgent(learning_rate=learning_rates[1],
                num_actions=num_primitive_actions,
                state_dims=[state_sizes[1] + num_subgoals],
                epsilon_end=0.01)

        self._subgoals = subgoals
        self._num_subgoals = num_subgoals

        self._meta_controller_state_fn = meta_controller_state_fn
        self._check_subgoal_fn = check_subgoal_fn

        self._meta_controller_state = None
        self._next_meta_controller_state = None
        self._curr_subgoal = None
        self._meta_controller_reward = 0
        self._intrinsic_time_step = 0
        self._episode = 0

    def get_meta_controller_state(self, state):
        returned_state = state
        if self._meta_controller_state_fn:
            returned_state = self._meta_controller_state_fn(state, self._original_state)

        return returned_state

    def get_controller_state(self, state, subgoal_index):
        curr_subgoal = self._subgoals[subgoal_index]

        # Concatenate the environment state with the subgoal.
        controller_state = list(state[0])
        for i in xrange(len(curr_subgoal)):
            controller_state.append(curr_subgoal[i])
        controller_state = np.array([controller_state])
        # print controller_state
        return np.copy(controller_state)

    def intrinsic_reward(self, state, subgoal_index):
        if self.subgoal_completed(state, subgoal_index):
            return 1
        else:
            return self.INTRINSIC_STEP_COST

    def subgoal_completed(self, state, subgoal_index):
        if self._check_subgoal_fn is None:
            return state == self._subgoals[subgoal_index]
        else:
            return self._check_subgoal_fn(state, subgoal_index)

    def store(self, state, action, reward, next_state, terminal, eval=False):
        """Stores the current transition in replay memory.
           The transition is stored in the replay memory of the controller.
           If the transition culminates in a subgoal's completion or a terminal state, a
           transition for the meta-controller is constructed and stored in its replay buffer.

           Args:
            state: current state
            action: primitive action taken
            reward: reward received from state-action pair
            next_state: next state
            terminal: extrinsic terminal (True or False)
            eval: Whether the current episode is a train or eval episode.
        """

        # Compute the controller state, reward, next state, and terminal.
        intrinsic_state = self.get_controller_state(state, self._curr_subgoal)
        intrinsic_next_state = self.get_controller_state(next_state, self._curr_subgoal)
        intrinsic_reward = self.intrinsic_reward(next_state, self._curr_subgoal)
        subgoal_completed = self.subgoal_completed(next_state, self._curr_subgoal)
        intrinsic_terminal = subgoal_completed or terminal

        # Store the controller transition in memory.
        self._controller.store(np.copy(intrinsic_state), action,
            intrinsic_reward, np.copy(intrinsic_next_state), intrinsic_terminal, eval)

        self._meta_controller_reward += reward
        self._intrinsic_time_step += 1

        if terminal and not eval:
            self._episode += 1

        if subgoal_completed or terminal:

            meta_controller_state = np.copy(self._meta_controller_state)
            next_meta_controller_state = self.get_meta_controller_state(next_state)

            # Store the meta-controller transition in memory.
            self._meta_controller.store(np.copy(meta_controller_state), self._curr_subgoal,
                self._meta_controller_reward, np.copy(next_meta_controller_state),
                terminal, eval)

            # Reset the current meta-controller state and current subgoal to be None
            # since the current subgoal is finished. Also reset the meta-controller's reward.
            self._next_meta_controller_state = np.copy(next_meta_controller_state)

            if terminal:
                self._next_meta_controller_state = None

            self._meta_controller_state = None
            self._curr_subgoal = None
            self._meta_controller_reward = 0
            self._intrinsic_time_step = 0

    def sample(self, state):
        """Samples an action from the hierarchical DQN agent.
           Samples a subgoal if necessary from the meta-controller and samples a primitive action
           from the controller.

           Args:
            state: the current environment state.

           Returns:
            action: a primitive action.
        """
        if self._meta_controller_state is None:

            if self._next_meta_controller_state is not None:
                self._meta_controller_state = self._next_meta_controller_state
            else:
                self._meta_controller_state = self.get_meta_controller_state(state)

            self._curr_subgoal = self._meta_controller.sample([self._meta_controller_state])

        controller_state = self.get_controller_state(state, self._curr_subgoal)
        action = self._controller.sample(controller_state)

        return action

    def best_action(self, state):
        """Returns the greedy action from the hierarchical DQN agent.
           Gets the greedy subgoal if necessary from the meta-controller and gets
           the greedy primitive action from the controller.

           Args:
            state: the current environment state.

           Returns:
            action: the controller's greedy primitive action.
        """

        if self._meta_controller_state is None:

            if self._next_meta_controller_state is not None:
                self._meta_controller_state = self._next_meta_controller_state
            else:
                self._meta_controller_state = self.get_meta_controller_state(state)

            self._curr_subgoal = self._meta_controller.best_action([self._meta_controller_state])

        controller_state = self.get_controller_state(state, self._curr_subgoal)
        action = self._controller.best_action(controller_state)
        return action

    def update(self):
        self._controller.update()
        # Only update meta-controller right after a meta-controller transition has taken place,
        # which occurs only when either a subgoal has been completed or the agnent has reached a
        # terminal state.
        if self._meta_controller_state is None:
            self._meta_controller.update()
