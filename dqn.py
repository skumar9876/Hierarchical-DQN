import numpy as np
import random
from replay_buffer import ReplayBuffer
import tensorflow as tf

class DqnAgent(object):

    # Discount factor for future rewards.
    DISCOUNT = 0.99
    # Max size of the replay buffer.
    REPLAY_MEMORY_SIZE = 500000
    # Batch size for updates from the replay buffer.
    BATCH_SIZE = 32
    # Initial size of replay memory prior to beginning sampling batches.
    REPLAY_MEMORY_INIT_SIZE = 5000
    # Update the target network every TARGET_UPDATE timesteps.
    TARGET_UPDATE = 1000 #10000

    def __init__(self, sess=None, learning_rate=0.00025, state_dims=[], num_actions=0,
        epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=50000, replay_memory_init_size=None,
        target_update=None):

        self._learning_rate = learning_rate
        self._state_dims = state_dims
        self._num_actions = num_actions

        self._epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
        self._epsilon_decay_steps = epsilon_decay_steps

        if replay_memory_init_size is not None:
            self.REPLAY_MEMORY_INIT_SIZE = replay_memory_init_size

        if target_update is not None:
            self.TARGET_UPDATE = target_update

        self._replay_buffer = ReplayBuffer(
            self.REPLAY_MEMORY_SIZE,
            self.REPLAY_MEMORY_INIT_SIZE,
            self.BATCH_SIZE)

        self._current_time_step = 0

        with tf.Graph().as_default():
            self._construct_graph()
            self._saver = tf.train.Saver()
            if sess is None:
                self.sess = tf.Session()
            else:
                self.sess = sess
            self.sess.run(tf.global_variables_initializer())

    def _q_network(self, state):

        layer1 = tf.contrib.layers.fully_connected(state, 64, activation_fn=tf.nn.relu)
        q_values = tf.contrib.layers.fully_connected(layer1, self._num_actions, activation_fn=None)

        return q_values

    def _construct_graph(self):
        shape=[None]
        for dim in self._state_dims:
            shape.append(dim)
        self._state = tf.placeholder(shape=shape, dtype=tf.float32)

        with tf.variable_scope('q_network'):
            self._q_values = self._q_network(self._state)
        with tf.variable_scope('target_q_network'):
            self._target_q_values = self._q_network(self._state)
        with tf.variable_scope('q_network_update'):
            self._picked_actions = tf.placeholder(shape=[None, 2], dtype=tf.int32)
            self._td_targets = tf.placeholder(shape=[None], dtype=tf.float32)
            self._q_values_pred = tf.gather_nd(self._q_values, self._picked_actions)
            self._losses = clipped_error(self._q_values_pred - self._td_targets)
            self._loss = tf.reduce_mean(self._losses)

            self.optimizer = tf.train.RMSPropOptimizer(self._learning_rate)

            grads_and_vars = self.optimizer.compute_gradients(self._loss, tf.trainable_variables())

            grads = [gv[0] for gv in grads_and_vars]
            params = [gv[1] for gv in grads_and_vars]
            grads = tf.clip_by_global_norm(grads, 5.0)[0]

            clipped_grads_and_vars = zip(grads, params)
            self.train_op = self.optimizer.apply_gradients(clipped_grads_and_vars,
                global_step=tf.contrib.framework.get_global_step())

        with tf.name_scope('target_network_update'):
            q_network_params = [t for t in tf.trainable_variables() if t.name.startswith(
                'q_network')]
            q_network_params = sorted(q_network_params, key=lambda v: v.name)

            target_q_network_params = [t for t in tf.trainable_variables() if t.name.startswith(
                'target_q_network')]
            target_q_network_params = sorted(target_q_network_params, key=lambda v: v.name)

            self.target_update_ops = []
            for e1_v, e2_v in zip(q_network_params, target_q_network_params):
                op = e2_v.assign(e1_v)
                self.target_update_ops.append(op)

    def sample(self, state):
        self._current_time_step += 1
        q_values = self.sess.run(self._q_values, {self._state: state})

        epsilon = self._epsilons[min(self._current_time_step, self._epsilon_decay_steps - 1)]

        e = random.random()
        if e < epsilon:
            return random.randint(0, self._num_actions - 1)
        else:
            return np.argmax(q_values)

    def best_action(self, state):
        q_values = self.sess.run(self._q_values, {self._state: state})
        return np.argmax(q_values)

    def store(self, state, action, reward, next_state, terminal, eval=False, curr_reward=False):
        if not eval:
            self._replay_buffer.add(state, action, reward, next_state, terminal)

    def update(self):
        states, actions, rewards, next_states, terminals = self._replay_buffer.sample()
        actions = zip(np.arange(len(actions)), actions)

        if len(states) > 0:
            next_states_q_values = self.sess.run(self._target_q_values, {self._state: next_states})
            next_states_max_q_values = np.max(next_states_q_values, axis=1)
            td_targets = rewards + (1 - terminals) * self.DISCOUNT * next_states_max_q_values

            feed_dict = {self._state: states,
                         self._picked_actions: actions,
                         self._td_targets: td_targets}

            _ = self.sess.run(self.train_op, feed_dict=feed_dict)

        # Update the target q-network.
        if not self._current_time_step % self.TARGET_UPDATE:
            self.sess.run(self.target_update_ops)

def clipped_error(x):
    return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

def compute_gradients(tensor, var_list):
  grads = tf.gradients(tensor, var_list)
  return [grad if grad is not None else tf.zeros_like(var)
          for var, grad in zip(var_list, grads)]
