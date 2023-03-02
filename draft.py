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
    a = torch.arange(0, 10).reshape((10, 1))        # 10 1
    b = torch.tensor([-1, 2, 0, 3, 4, 5, 10, 12, 4, 9]).reshape((10, 1))    #10 1
    c = torch.minimum(a, b)     # 10 1
    d = torch.ones(10)  # 10
    # print(a.shape)
    # print(b.shape)
    # print(c.shape)
    # print(d.shape)
