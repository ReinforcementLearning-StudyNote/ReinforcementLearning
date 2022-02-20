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
# import emoji
import turtle


if __name__ == '__main__':
    a = torch.ones((10, 6))
    b = torch.split(a, [4,6], dim=0)
    print(b[1])
    # print(a1)
    # print(a2)
    pass
