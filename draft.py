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

import multiprocessing
import numpy as np
import time

def write_to_shared_memory(shared_mem):
    # 写入数据到共享内存
    arr = np.zeros((100,100))
    for i in range(100):
        for j in range(100):
            arr[i][j] = i + j
    shared_mem_np = np.ndarray(shared_mem.shape, dtype=np.float32, buffer=shared_mem.buf)
    shared_mem_np[:] = arr[:]

def read_from_shared_memory(shared_mem):
    # 从共享内存中读取数据
    shared_mem_np = np.ndarray(shared_mem.shape, dtype=np.float32, buffer=shared_mem.buf)
    for i in range(100):
        for j in range(100):
            print(shared_mem_np[i][j], end=' ')
        print()

if __name__ == '__main__':
    # 创建共享内存
    shared_mem = multiprocessing.shared_memory.SharedMemory(create=True, size=100*100*4)

    # 创建两个进程，一个写入数据到共享内存，一个读取数据
    p1 = multiprocessing.Process(target=write_to_shared_memory, args=(shared_mem,))
    p2 = multiprocessing.Process(target=read_from_shared_memory, args=(shared_mem,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    # 删除共享内存
    shared_mem.close()
    shared_mem.unlink()

