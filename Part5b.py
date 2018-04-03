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
from Part5a import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys

class Policy(nn.Module):
    """
    The Tic-Tac-Toe Policy
    """
    def __init__(self, input_size=27, hidden_size=64, output_size=9):
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


if __name__ == '__main__': 
	env = Environment()
	#policy = Policy(hidden_size=64)
	total_episodes = 50000
	track = []

	if len(sys.argv) == 1:
	    # `python tictactoe.py` to train the agent
	    for hidden in [600,500,400,300]:
	    	episode_track, average_track = train(Policy(hidden_size=hidden), env, total_episodes)
	    	track.append([episode_track, average_track])

	else:
	    # `python tictactoe.py <ep>` to print the first move distribution
	    # using weightt checkpoint at episode int(<ep>)
	    ep = int(sys.argv[1])
	    load_weights(policy, ep)
	    print(first_move_distr(policy, env))

	plt.plot(track[0][0], track[0][1], 'r', \
			track[1][0], track[1][1], 'b', \
			track[2][0], track[2][1], 'g', \
			track[3][0], track[3][1], 'y')
	red_patch = mpatches.Patch(color='red', label='600')
	blue_patch = mpatches.Patch(color='blue', label='500')
	green_patch = mpatches.Patch(color='green', label='400')
	yellow_patch = mpatches.Patch(color='yellow', label='300')
	plt.legend(handles=[red_patch, blue_patch, green_patch, yellow_patch])
	plt.xlabel('Episodes')
	plt.ylabel('Average Return')
	plt.title('Training Curve of the TicTacToe Model')
	plt.show()