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
    a = np.array([1,2])
    b = np.array([3,4])
    c = np.array([-1,5])
    print(np.max(a))

