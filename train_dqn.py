"""
@author: Saurabh Kumar
"""

import os

import matplotlib
matplotlib.use('Agg')

import clustering
import dqn
import gym
from gym.wrappers import Monitor
import hierarchical_dqn
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle

tf.flags.DEFINE_string('agent_type', 'h_dqn', 'RL agent type.')
tf.flags.DEFINE_string('logdir', 'experiment_logs/', 'Directory of logfile.')
tf.flags.DEFINE_string('experiment_dir', '', 'Directory of experiment files.')
tf.flags.DEFINE_string('logfile', 'log.txt', 'Name of the logfile.')
tf.flags.DEFINE_string('env_name', 'MountainCar-v0', 'Name of the environment.')

env_name = ''

FLAGS = tf.flags.FLAGS


def log(logfile, iteration, rewards):
    """Function that logs the reward statistics obtained by the agent.

    Args:
        logfile: File to log reward statistics.
        iteration: The current iteration.
        rewards: Array of rewards obtained in the current iteration.
    """
    log_string = '{} {} {} {}'.format(
        iteration, np.min(rewards), np.mean(rewards), np.max(rewards))
    print(log_string)

    with open(logfile, 'a') as f:
        f.write(log_string + '\n')


def make_environment(env_name):
    return gym.make(env_name)


def make_agent(agent_type, env, num_clusters, use_extra_travel_penalty, use_extra_bit,
    use_controller_dqn, use_intrinsic_timeout, use_memory, memory_size, pretrain_controller):
    if agent_type == 'dqn':
        return dqn.DqnAgent(state_dims=[2],
                            num_actions=2) # env.action_space.n
    elif agent_type == 'h_dqn':
        meta_controller_state_fn, check_subgoal_fn, num_subgoals, subgoals = clustering.get_cluster_fn(
            n_clusters=num_clusters, extra_bit=use_extra_bit)

        return hierarchical_dqn.HierarchicalDqnAgent(
            state_sizes=[num_subgoals, [2]],
            agent_types=['tabular', 'network'],
            subgoals=subgoals,
            num_subgoals=num_subgoals,
            num_primitive_actions=2, # env.action_space.n
            meta_controller_state_fn=meta_controller_state_fn,
            check_subgoal_fn=check_subgoal_fn,
            use_extra_travel_penalty=use_extra_travel_penalty,
            use_extra_bit_for_subgoal_center=use_extra_bit,
            use_controller_dqn=use_controller_dqn,
            use_intrinsic_timeout=use_intrinsic_timeout,
            use_memory=use_memory,
            memory_size=memory_size,
            pretrain_controller=pretrain_controller)


def run(env_name='MountainCar-v0',
        agent_type='dqn',
        num_iterations=1000,
        num_train_episodes=100,
        num_eval_episodes=100,
        logdir=None,
        experiment_dir=None,
        logfile=None):
    """Function that executes RL training and evaluation.

    Args:
        env_name: Name of the environment that the agent will interact with.
        agent_type: The type RL agent that will be used for training.
        num_iterations: Number of iterations to train for.
        num_train_episodes: Number of training episodes per iteration.
        num_eval_episodes: Number of evaluation episodes per iteration.
        logdir: Directory for log file.
        logfile: File to log the agent's performance over training.
    """
    experiment_dir += '_agent_type_' + agent_type

    experiment_dir = logdir + experiment_dir
    logfile = experiment_dir + '/' + logfile

    try:
        os.stat(experiment_dir)
    except:
        os.mkdir(experiment_dir)


    env = make_environment(env_name)
    env_test = make_environment(env_name)
    # env_test = Monitor(env_test, directory='videos/', video_callable=lambda x: True, resume=True)
    print 'Made environment!'
    agent = make_agent(agent_type, env)
    print 'Made agent!'

    for it in range(num_iterations):

        # Run train episodes.
        for train_episode in range(num_train_episodes):
            # Reset the environment.
            state = env.reset()
            state = np.expand_dims(state, axis=0)

            episode_reward = 0

            # Run the episode.
            terminal = False

            while not terminal:
                action = agent.sample(state)
                # Remove the do-nothing action.
                if env_name == 'MountainCar-v0':
                    if action == 1:
                        env_action = 2
                    else:
                        env_action = action

                next_state, reward, terminal, _ = env.step(env_action)
                next_state = np.expand_dims(next_state, axis=0)

                agent.store(state, action, reward, next_state, terminal)
                agent.update()

                episode_reward += reward
                # Update the state.
                state = next_state

        eval_rewards = []

        # Run eval episodes.
        for eval_episode in range(num_eval_episodes):

            # Reset the environment.
            state = env_test.reset()
            state = np.expand_dims(state, axis=0)

            episode_reward = 0

            # Run the episode.
            terminal = False

            while not terminal:
    		if agent_type == 'dqn':
    		    action = agent.best_action(state)
                else:
                    action, info = agent.best_action(state)
                if agent_type == 'h_dqn' and info is not None:
                    curr_state = info[0]
                    if not use_memory:
                        curr_state = np.where(np.squeeze(curr_state) == 1)[0][0]
                    else:
                        curr_state = np.squeeze(curr_state)[-1] - 1
                    goal = info[1]
                    heat_map[curr_state][goal] += 1

                # Remove the do-nothing action.
                if action == 1:
                    env_action = 2
                else:
                    env_action = action

                next_state, reward, terminal, _ = env_test.step(env_action)

                next_state = np.expand_dims(next_state, axis=0)
                # env_test.render()
                agent.store(state, action, reward, next_state, terminal, eval=True)
                if reward > 1:
                    reward = 1 # For sake of comparison.

                episode_reward += reward

                state = next_state

            eval_rewards.append(episode_reward)

        with open(experiment_dir + '/eval_rewards_' + str(it), 'wb') as f:
            pickle.dump(eval_rewards, f)

        log(logfile, it, eval_rewards)


run(agent_type=FLAGS.agent_type, logdir=FLAGS.logdir, experiment_dir=FLAGS.experiment_dir,
    logfile=FLAGS.logfile)


