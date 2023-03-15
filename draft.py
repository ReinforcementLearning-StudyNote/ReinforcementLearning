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


# a = torch.tensor([[0.1, 0.4, 0.4, 0.1], [0.25, 0.25, 0.25, 0.25]])
# b = a.multinomial(10000, True)
# print(a.shape)
# print(b.shape)
# print(b)

x = torch.FloatTensor([1, 2, 3, 4])
x1 = (x[0] * x[1]).unsqueeze(dim=0)
x2 = (x[1]).unsqueeze(dim=0)
x3 = (x[2]).unsqueeze(dim=0)
x4 = (x[3]).unsqueeze(dim=0)
y = torch.cat((x1, x2, x3, x4)).requires_grad_(True)
# print(y.shape)
# print(y.requires_grad)
print(np.fabs(np.array([-1,-23,4,-5])))