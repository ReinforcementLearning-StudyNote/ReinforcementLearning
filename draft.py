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
    # print(emoji.emojize('Python is :thumbs_up:'))
    # # Python is 👍
    # print(emoji.emojize('Python is :thumbsup:', use_aliases=True))
    # # Python is 👍
    # print(emoji.demojize('Python is 👍'))
    # # Python is :thumbs_up:
    # print(emoji.emojize("Python is fun :red_heart:"))
    # # Python is fun ❤
    # print(emoji.emojize("Python is fun :red_heart:", variant="emoji_type"))
    print(emoji.emojize('Lanxin~ Happy New Year~ :green_heart:', variant="emoji_type"))
    pass
