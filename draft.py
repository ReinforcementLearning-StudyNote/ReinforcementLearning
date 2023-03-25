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

a = np.arange(2)
share_a = shared_memory.SharedMemory(create=True, size=a.nbytes)

def send(sh:shared_memory.SharedMemory):
	sh2 = shared_memory.SharedMemory(name=sh.name)
	c = np.ndarray(a.shape, a.dtype, sh2.buf)
	try:
		while True:
			time.sleep(0.2)
			c[0] += 1
			c[1] += 1
	except:
		sh2.close()

def receive(sh:shared_memory.SharedMemory):
	sh1 = shared_memory.SharedMemory(name=sh.name)
	b = np.ndarray(a.shape, a.dtype, sh1.buf)
	try:
		while True:
			print(b)
			time.sleep(0.2)
	except:
		sh1.close()

p1 = mp.Process(target=send, args=(share_a,))
p2 = mp.Process(target=receive, args=(share_a,))
p1.start()
p2.start()
p1.join()
p2.join()
@atexit.register
def clean():
	share_a.close()
	share_a.unlink()
