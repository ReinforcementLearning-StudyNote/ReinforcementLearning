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

a = torch.ones((10, 10))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
