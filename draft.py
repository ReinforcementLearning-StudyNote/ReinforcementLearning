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

def add(a):
    for i in [0,1,2,3]:
        a[i] += 1

if __name__ == '__main__':
    x = np.linspace(-10, 10, 10000)
    y1 = np.tanh(x)
    k = 0.5
    y2 = (np.exp(k * x) - 1) / (np.exp(k * x) + 1)
    plt.plot(x,y1)
    plt.plot(x, y2)
    plt.grid(True)
    plt.show()
    pass
