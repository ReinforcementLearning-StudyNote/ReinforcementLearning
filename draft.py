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
import emoji

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def add(a):
    for i in [0,1,2,3]:
        a[i] += 1

if __name__ == '__main__':
    a = torch.randint(low=-10,high=11,size=[1, 10]) / 10
    b = torch.sign(a)
    c = torch.clamp(input=b, min=0, max=1)
    print(c.size())
    print(a)
    print(b)
    print(c)
    # b = torch.ones([1, 10])
    # c = torch.randint(low=1, high=10, size=[1,10])
    # d = torch.cat((b, c), dim=0)
    # e = torch.mul(a, d)
    # print(a)
    # print(e)
    pass
