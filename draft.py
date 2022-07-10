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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import random
    from pylab import mpl

    # 设置显示中文字体
    mpl.rcParams["font.sans-serif"] = ["SimHei"]

    # 0.准备数据
    # x = np.random.uniform(0, 1, 1000000)  # 在[0,1)上产生100000000个数
    x = np.random.beta(a=2, b=2, size=1000000)

    # 1.创建画布
    # plt.figure(figsize=(10, 10), dpi=100)

    plt.xlim(0,1)
    plt.hist(x, bins=100)  # 0-1区间上 划分为1000份
    # 2.1 保存图像

    # 3.图像展示
    plt.show()
