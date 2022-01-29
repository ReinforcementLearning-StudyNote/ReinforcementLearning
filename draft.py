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
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def add(a):
    for i in [0,1,2,3]:
        a[i] += 1

if __name__ == '__main__':
    # points = np.random.uniform(low=0.3, high=10, size=20).reshape((-1, 2))
    # # print(points)
    # bezier = Bezier(points)
    # a = bezier.Curve()
    # # bezier.Draw()
    # print(a)
    # a = np.linspace(0, 4,5)
    # b = np.linspace(5, 9, 5)
    # c = np.array([a, b])
    # print(a, b)
    # print(c.T)
    a = np.linspace(0, 10,11)
    print(a[0:])
    # print(np.clip(a, -10, 2))
    pass
