import os
import time
import numpy as np
import math
from environment.config.xml_write import xml_cfg
import random
from common.common import *
import torch

# a = [1, 1]
# b = [-1, 1]
# cos = np.dot(a,b) / np.linalg.norm(a) / np.linalg.norm(b)
# print(cos)
# print(np.arccos(cos))
a = torch.randn(10, 4)
b = torch.zeros(10)
print(a,b)
maxa = torch.max(input=a, dim=1, keepdim=True)
print(maxa[0].mul(b))
