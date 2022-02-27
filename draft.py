import os
import time
import numpy as np
import math
from environment.config.xml_write import xml_cfg
import random
import cv2 as cv
from common.common import *
import torch
import pandas as pd
import matplotlib.pyplot as plt
from environment.envs.pathplanning.bezier import Bezier


if __name__ == '__main__':
    # clip = 1/3
    # a = GaussianNoise(mu=np.zeros(1))
    # b = torch.clip(torch.tensor(a(sigma=1.0)), -clip, clip)
    # print(b)
    # print(random.randint(0, 1))
    a = torch.rand((10000, 10000))
    b = torch.rand((10000, 10000))
    print('Size a:', a.size())
    print('Size b:', b.size())
    t0 = time.time()
    for _ in range(100):
        c = torch.matmul(a, b)
    t1 = time.time()
    print('Device:', a.device, '  Time:', t1 - t0)

    print('Start moving matrix to GPU...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()
    a = a.to(device)
    b = b.to(device)
    t1 = time.time()
    print('time:', t1 - t0)
    print('Finish moving matrix to GPU...')

    t0 = time.time()
    for _ in range(100):
        c = torch.matmul(a, b)
    t1 = time.time()

    print('Device:', a.device, '  Time:', t1 - t0)
    pass
