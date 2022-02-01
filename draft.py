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
    a = [0, 1, 2, 3, 4, 5, 6]
    for i in range(7):
        print(a[i], a[6-i])
    pass
