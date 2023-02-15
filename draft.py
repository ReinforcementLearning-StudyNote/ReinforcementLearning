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


if __name__ == '__main__':
    # a = np.array([1, 2, 3])
    # b = np.diag([1, -1, 2])
    # print(np.dot(a, [1, -1, 1]))
    # a = np.array([1,2])
    # b = np.array([3,4])
    # c = np.array([-1,5])
    # print(np.max(a))
    # psi = deg2rad(90)
    # theta = deg2rad(82)
    # phi = deg2rad(23)
    # a = (math.cos(psi) * math.sin(theta) * math.cos(phi) + math.sin(psi) * math.sin(phi)) ** 2
    # b = (math.sin(psi) * math.sin(theta) * math.cos(phi) - math.cos(psi) * math.sin(phi)) ** 2
    # c = (math.cos(phi) * math.cos(theta)) ** 2
    # print(a + b + c)
    a = np.array([1, 2, 3, 4, 5])
    for i in a:
        if i > 3:
            i -=1
        else:
            i+=1
    print(a)