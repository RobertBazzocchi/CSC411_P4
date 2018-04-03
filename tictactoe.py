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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

global Invalid_reward
Invalid_reward = -150

class Environment(object):
    """
    The Tic-Tac-Toe Environment
    """
    # possible ways to win
    win_set = frozenset([(0,1,2), (3,4,5), (6,7,8), # horizontal
                         (0,3,6), (1,4,7), (2,5,8), # vertical
                         (0,4,8), (2,4,6)])         # diagonal
    # statuses
    STATUS_VALID_MOVE = 'valid'
    STATUS_INVALID_MOVE = 'inv'
    STATUS_WIN = 'win'
    STATUS_TIE = 'tie'
    STATUS_LOSE = 'lose'
    STATUS_DONE = 'done'

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the game to an empty board."""
        self.grid = np.array([0] * 9) # grid
        self.turn = 1                 # whose turn it is
        self.done = False             # whether game is doneT
        return self.grid

    def render(self):
        """Print what is on the board."""
        map = {0:'.', 1:'x', 2:'o'} # grid label vs how to plot
        print(''.join(map[i] for i in self.grid[0:3]))
        print(''.join(map[i] for i in self.grid[3:6]))
        print(''.join(map[i] for i in self.grid[6:9]))
        print('====')

    def check_win(self):
        """Check if someone has won the game."""
        for pos in self.win_set:
            s = set([self.grid[p] for p in pos])
            if len(s) == 1 and (0 not in s):
                return True
        return False

    def step(self, action):
        """Mark a point on position action."""
        assert type(action) == int and action >= 0 and action < 9
        # done = already finished the game
        if self.done:
            return self.grid, self.STATUS_DONE, self.done
        # action already have something on it
        if self.grid[action] != 0:
            return self.grid, self.STATUS_INVALID_MOVE, self.done
        # play move
        self.grid[action] = self.turn
        if self.turn == 1:
            self.turn = 2
        else:
            self.turn = 1
        # check win
        if self.check_win():
            self.done = True
            return self.grid, self.STATUS_WIN, self.done
        # check tie
        if all([p != 0 for p in self.grid]):
            self.done = True
            return self.grid, self.STATUS_TIE, self.done
        return self.grid, self.STATUS_VALID_MOVE, self.done

    def random_step(self):
        """Choose a random, unoccupied move on the board to play."""
        pos = [i for i in range(9) if self.grid[i] == 0]
        move = random.choice(pos)
        return self.step(move)

    def play_against_random(self, action, display_game=False):
        """Play a move, and then have a random agent play the next move."""
        state, status, done = self.step(action)
        if display_game: env.render()
        if not done and self.turn == 2:
            state, s2, done = self.random_step()
            if display_game: env.render()
            if done:
                if s2 == self.STATUS_WIN:
                    status = self.STATUS_LOSE
                elif s2 == self.STATUS_TIE:
                    status = self.STATUS_TIE
                else:
                    raise ValueError("???")
        return state, status, done

class Policy(nn.Module):
    """
    The Tic-Tac-Toe Policy
    """
    def __init__(self, input_size=27, hidden_size=500, output_size=9):
        super(Policy, self).__init__()

        ######### CODE ADDED BELOW ######### 

        # NOTES:
        # - The Python super() function creates an instance of the base nn.Module class
        #
        # - A fully connected neural network layer is represented by the nn.Linear object,
        #   with the first argument in the definition being the number of nodes in layer l
        #   and the next argument being the number of nodes in layer l+1

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):

        # NOTES:
        # - Now that we’ve setup the “skeleton” of our network architecture, we have to
        #   define how data flows through out network.
        # - We apply a ReLu activation on the first layer and a softmax activation on the
        #   second. This, combined with the negative log likelihood loss function gives us a
        #   multi-class cross entropy based loss function which we will use to train the network.

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # RETURNS LOG PROB, BUT CAUSED PROBLEMS IN select_action (DID NOT ACCEPT NEGATIVE PROBABILITIES)
        # return F.log_softmax(x,dim=1))
        return F.softmax(x,dim=1) # no dim arguments outputs the same with a UserWarning

def select_action(policy, state):
    """Samples an action from the policy at the state."""
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    pr = policy(Variable(state))
    m = torch.distributions.Categorical(pr)
    action = m.sample()
    log_prob = torch.sum(m.log_prob(action))
    return action.data[0], log_prob

def compute_returns(rewards, gamma=1.0):
    """
    Compute returns for each time step, given the rewards
      @param rewards: list of floats, where rewards[t] is the reward
                      obtained at time step t
      @param gamma: the discount factor
      @returns list of floats representing the episode's returns
          G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... 
    """
    n = 0
    G = []
    while n < len(rewards):
        G.append(0)  
        for i in range(n,len(rewards)):
            G[n] += (gamma**(i-n))*rewards[i]
        n+=1

    return G

def finish_episode(saved_rewards, saved_logprobs, gamma=1.0):
    """Samples an action from the policy at the state."""
    policy_loss = []
    returns = compute_returns(saved_rewards, gamma)
    returns = torch.Tensor(returns)
    # subtract mean and std for faster training
    returns = (returns - returns.mean()) / (returns.std() +
                                            np.finfo(np.float32).eps)
    for log_prob, reward in zip(saved_logprobs, returns):
        policy_loss.append(-log_prob * reward)
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward(retain_graph=True)
    # note: retain_graph=True allows for multiple calls to .backward()
    # in a single step

def get_reward(status):
    """Returns a numeric given an environment status."""

    return {                   
            Environment.STATUS_VALID_MOVE  :    0, 
            Environment.STATUS_INVALID_MOVE: Invalid_reward, #-150, # -100
            Environment.STATUS_WIN         :  100,
            Environment.STATUS_TIE         :    0,   
            Environment.STATUS_LOSE        : -200  # -200
    }[status]


def train(policy, env, gamma=0.75, log_interval=1000):
    """Train policy gradient."""
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10000, gamma=0.9)
    running_reward = 0

    num_invalid_moves = 0
    num_wins = 0
    num_losses = 0
    num_ties = 0

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
            if len(saved_rewards) > 1000:
                break
            

        if status == "win":
            num_wins += 1
        if status == "lose":
            num_losses += 1
        if status == "tie":
            num_ties += 1
        
        for reward in saved_rewards:
            if reward == Invalid_reward:
                num_invalid_moves += 1

        R = compute_returns(saved_rewards)[0]
        running_reward += R

        finish_episode(saved_rewards, saved_logprobs, gamma)

        if i_episode % log_interval == 0:
            print('Episode {}\tAverage return: {:.2f}'.format(
                i_episode,
                running_reward / log_interval))
            print("Wins: {}, Losses: {}, Ties: {}, Invalid Moves: {}\n".format(num_wins,num_losses,num_ties,num_invalid_moves))
            num_wins,num_losses,num_ties,num_invalid_moves = 0,0,0,0
            running_reward = 0

        if i_episode % (log_interval) == 0:
            torch.save(policy.state_dict(),
                       "ttt/policy-%d.pkl" % i_episode)

        if i_episode % 1 == 0: # batch_size
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # STOP AFTER 50000 ITERATIONS
        if i_episode == 50000:
            break

def first_move_distr(policy, env):
    """Display the distribution of first moves."""
    state = env.reset()
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    pr = policy(Variable(state))
    return pr.data

def load_weights(policy, episode):
    """Load saved weights"""
    weights = torch.load("ttt/policy-%d.pkl" % episode)
    policy.load_state_dict(weights)

def test(policy,env):
    """ 
    Test the policy on 100 games and output the wins, losses, and ties for all games.
    Additionally, display 5 of these games.
    """
    wins = 0
    losses = 0
    ties = 0
    num_invalid_moves = 0
    game_num = 1


    n = 100
    while n > 0:
        state = env.reset()
        done = False
        display_game = False

        if n % 20 == 0:
            display_game = True
            print("________________________________")
            print("GAME {} DISPLAYED".format(game_num))
            game_num += 1
            env.render()

        saved_rewards = []
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action,display_game)
            reward = get_reward(status)
            saved_rewards.append(reward)

        if status == "win":
            wins += 1
        if status == "lose":
            losses += 1
        if status == "tie":
            ties += 1

        for reward in saved_rewards:
            if reward == Invalid_reward:
                num_invalid_moves += 1

        n -= 1


    print("Wins: {}, Losses: {}, Ties: {}, Invalid: {}" \
        .format(wins,losses,ties,num_invalid_moves))
    return wins, losses, ties

if __name__ == '__main__':
    import sys
    env = Environment()
    policy = Policy()

    if len(sys.argv) == 1:
        # `python tictactoe.py` to train the agent
        train(policy, env)
        test(policy,env)

    else:
        # `python tictactoe.py <ep>` to print the first move distribution
        # using weightt checkpoint at episode int(<ep>)
        ep = int(sys.argv[1])
        load_weights(policy, ep)
        print(first_move_distr(policy, env))

"""
Use your learned policy to play 100 games against random. How many did your agent 
win / lose / tie? Display five games that your trained agent plays against the random policy.
Explain any strategies that you think your agent has learned.
"""
