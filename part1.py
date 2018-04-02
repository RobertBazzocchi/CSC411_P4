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
import sys
from tictactoe import *


def Part1():
    """
    FUNCTION:
        Plays a hard-coded game of tic-tac-toe and writes the visualized game state of each step
        to a .txt file. 
    """
    env = Environment()

    orig_stdout = sys.stdout
    f = open('out.txt', 'w')
    sys.stdout = f
    
    env.render()
    env.step(0)
    env.render()
    env.step(4)
    env.render()
    env.step(1)
    env.render()
    env.step(2)
    env.render()
    env.step(6)
    env.render()
    env.step(5)
    env.render()
    env.step(3)
    env.render()

    if env.turn == 1:
        winner = 2
    else:
        winner = 1

    print("Player {} has won the game.".format(winner))

    sys.stdout = orig_stdout
    f.close()

if __name__ == '__main__':
    Part1()