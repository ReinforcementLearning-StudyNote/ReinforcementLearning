import os
import time
import numpy as np
import math
from environment.config.xml_write import xml_cfg
import random
import cv2 as cv
from common.common_func import *
import torch
import pandas as pd
import matplotlib.pyplot as plt
from environment.envs.pathplanning.bezier import Bezier
import gym
import torch.nn as nn
import collections
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter


# dist = Normal(0, 0.1)
# for i in range(1000):
# 	writer.add_scalar('x', 2*np.sin(i*np.pi/100), i)	 # dist.sample()
# writer.close()
