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
    a = torch.ones((5,5))
    # b = a.clone()
    # b[0, 0] = 0
    print(a.requires_grad)

    # print(a.shape)
    # print(b.shape)
    # print(c.shape)
    # print(d.shape)
