from __future__ import print_function
from collections import defaultdict
from itertools import count
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions
from torch.autograd import Variable
from tictactoe import *
import matplotlib.pyplot as plt
import sys

def train(policy, env, max_episode, gamma=0.75, log_interval=1000):
    """Train policy gradient."""
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10000, gamma=0.9)
    running_reward = 0
    num_invalid_moves = 0

    episode_track = []
    average_track = []

    for i_episode in count(1):
        saved_rewards = []
        saved_logprobs = []
        state = env.reset()
        done = False

        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action)
            reward = get_reward(status)
            saved_logprobs.append(logprob)
            saved_rewards.append(reward)

        R = compute_returns(saved_rewards)[0]
        running_reward += R

        finish_episode(saved_rewards, saved_logprobs, gamma)

        if i_episode % log_interval == 0:
            print('Episode {}\tAverage return: {:.2f}'.format(
                i_episode,
                running_reward / log_interval))
            print("Number of Invalid Moves in Last {} Episodes: ".format(log_interval), num_invalid_moves)
            episode_track.append(i_episode)
            average_track.append(running_reward / log_interval)
            num_invalid_moves = 0
            running_reward = 0
            

        if i_episode % (log_interval) == 0:
            torch.save(policy.state_dict(),
                       "ttt/policy-%d.pkl" % i_episode)

        if i_episode % 1 == 0: # batch_size
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if i_episode == max_episode:
            break

    return episode_track, average_track

if __name__ == '__main__': 
    env = Environment()
    policy = Policy()
    total_episodes = 20000

    if len(sys.argv) == 1:
        # `python tictactoe.py` to train the agent
        episode_track, average_track = train(policy, env, total_episodes)

    else:
        # `python tictactoe.py <ep>` to print the first move distribution
        # using weightt checkpoint at episode int(<ep>)
        ep = int(sys.argv[1])
        load_weights(policy, ep)
        print(first_move_distr(policy, env))

    plt.plot(episode_track, average_track)
    plt.xlabel('Episodes')
    plt.ylabel('Average Return')
    plt.title('Training Curve of the TicTacToe Model')
    plt.show()