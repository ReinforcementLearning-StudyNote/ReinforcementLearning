import os
import time
import numpy as np
import math
from environment.config.xml_write import xml_cfg
import random
from common.common import *
import torch

# import torch
# a=torch.tensor([[.1, .2, .3],
#                 [1.1, 1.2, 1.3],
#                 [2.1, 2.2, 2.3],
#                 [3.1, 3.2, 3.3]])
# b = torch.argmax(a, dim=1, keepdim=True)
# print(b)
# a = random.sample(range(1, 10), 9)
a = np.random.choice(20, 5)
print(type(a))