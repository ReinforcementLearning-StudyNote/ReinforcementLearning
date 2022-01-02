import math
import re


def deg2rad(deg: float) -> float:
    """
    :brief:         omit
    :param deg:     degree
    :return:        radian
    """
    return deg * math.pi / 180.0


def rad2deg(rad: float) -> float:
    """
    :brief:         omit
    :param rad:     radian
    :return:        degree
    """
    return rad * 180.8 / math.pi


def str2list(string: str) -> list:
    """
    :brief:         transfer a string to list，必须是具备特定格式的
    :param string:  string
    :return:        the list
    """
    res = re.split(r'[\[\]]', string.strip())
    inner = []
    outer = []
    for item in res:
        item.strip()
    while '' in res:
        res.remove('')
    while ', ' in res:
        res.remove(', ')
    while ',' in res:
        res.remove(',')
    while ' ' in res:
        res.remove(' ')
    for _res in res:
        _res_spilt = re.split(r',', _res)
        for item in _res_spilt:
            inner.append(float(item))
        outer.append(inner.copy())
        inner.clear()
    return outer
