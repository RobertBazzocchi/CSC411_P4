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

if __name__ == '__main__': 
	env = Environment()
	policy = Policy(hidden_size=500)
	total_episodes = 20000
	track = []

	if len(sys.argv) == 1:
	    # `python tictactoe.py` to train the agent
	    for rate in [0.9,0.8,0.7,0.6]:
	    	episode_track, average_track = train(policy, env, total_episodes, gamma=rate)
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
	red_patch = mpatches.Patch(color='red', label='0.9')
	blue_patch = mpatches.Patch(color='blue', label='0.8')
	green_patch = mpatches.Patch(color='green', label='0.7')
	yellow_patch = mpatches.Patch(color='yellow', label='0.6')
	plt.legend(handles=[red_patch, blue_patch, green_patch, yellow_patch])
	plt.xlabel('Episodes')
	plt.ylabel('Average Return')
	plt.title('Training Curve of the TicTacToe Model with different gamma')
	plt.show()