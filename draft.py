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
    for _ in range(100):
        # a = np.random.rand(4, 2) * 10
        a = np.array([[5, 2],
                      [5, 8],
                      [6, 4],
                      [6, 10]])
        have, pt = cross_2_line_seg(a[0], a[1], a[2], a[3])
        plt.figure(0)
        plt.plot(a[0:2, 0], a[0:2, 1], c='red')
        plt.plot(a[2:4, 0], a[2:4, 1], c='blue')
        plt.scatter(a[0:2, 0], a[0:2, 1], c='red')
        plt.scatter(a[2:4, 0], a[2:4, 1], c='blue')
        if have:
            plt.scatter(pt[0], pt[1], c='black')
        plt.show()
