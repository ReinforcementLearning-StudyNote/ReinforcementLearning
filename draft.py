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



if __name__ == '__main__':
    fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    #
    # ax.set_xlabel('Time')
    # ax.set_ylabel('cos(t)')
    # ax.set_title('')

    line = None
    plt.grid(True)  # 添加网格
    plt.ion()  # interactive mode on
    # obsX = []
    # obsY = []
    #
    # t0 = time.time()
    xs = [0, 0]
    ys = [1, 1]
    for i in range(10000):
        y = np.random.random()
        xs[0] = xs[1]
        ys[0] = ys[1]
        xs[1] = i
        ys[1] = y
        # plt.plot(xs, ys)
        plt.plot(xs, ys, 'r')
        plt.pause(0.00001)

    pass
