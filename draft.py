import os
import time
import numpy as np
import math
from environment.config.xml_write import xml_cfg
import random
from common.common import *
import torch
import pandas as pd


# a = torch.tensor([1,2,3,4,4]).detach().numpy()
# print(np.argmax(a))
# rad_bet_pos_vel = np.arccos(2)
# print(rad_bet_pos_vel)
# quick_sort 代码实现

# def partition(arr, low: int, high: int):
#     pivot, j = arr[low], low
#     for i in range(low + 1, high + 1):
#         if arr[i] <= pivot:
#             j += 1
#             arr[j], arr[i] = arr[i], arr[j]
#     arr[low], arr[j] = arr[j], arr[low]
#     return j
#
#
# def quick_sort_between(arr, low: int, high: int):
#     if high - low <= 1:  # 递归结束条件
#         return
#
#     m = partition(arr, low, high)  # arr[m] 作为划分标准
#     quick_sort_between(arr, low, m - 1)
#     quick_sort_between(arr, m + 1, high)
#
#
# def quick_sort(arr):
#     """
#     快速排序(in-place)
#     :param arr: 待排序的List
#     :return: 快速排序是就地排序(in-place)
#     """
#     quick_sort_between(arr, 0, len(arr) - 1)


# 测试数据

if __name__ == '__main__':
    # # import random
    # # random.seed(54)
    # a = [random.randint(0,10) for _ in range(5)]
    # # print("原始数据：", a)
    # # quick_sort(arr)
    # # print("快速排序结果：", a)
    #
    # # df = pd.DataFrame({'A': [3, 4, 10, 23, 0, -2],
    # #                    'B': [2, 1, 9, 8, 7, 7],
    # #                    'C': [0, 1, 9, 4, 2, 8]})
    # # df = df.sort_values(by=['A'], ascending=True)
    # # print(df['B'])
    # sorted_id = sorted(range(len(a)), key=lambda k: a[k], reverse=False)
    # print(a)
    # print(a[-3:])
    # print(sorted_id)
    for _ in range(10):
        print(random.randint(0, 4))
    pass