import atexit
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
import torch.multiprocessing as mp
from multiprocessing import shared_memory

a = np.arange(10)
# print(a)
# # print(a.nbytes)
# print(size)
share_a = shared_memory.SharedMemory(create=True, name='yyf', size=a.nbytes)
b = np.ndarray(a.shape, a.dtype, share_a.buf)
c = np.ndarray(a.shape, a.dtype, share_a.buf)


# @atexit.register
# def clearn():
# 	share_a.close()
# 	share_a.unlink()
# 	print('close shared memory')
