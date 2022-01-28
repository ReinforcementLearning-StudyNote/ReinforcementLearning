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
    a = torch.arange(0, 10)
    print(a.dim(), a)
    b = a.unsqueeze(0)
    print(b.dim(), b)
    c = b.unsqueeze(0)
    print(c.dim(), c)
    pass
