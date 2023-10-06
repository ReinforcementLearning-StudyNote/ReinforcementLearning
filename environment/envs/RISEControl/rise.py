import numpy as np


class RISE:
    def __init__(self, c1: float = 0, c2: float = 0, k: float = 0, b: float = 0):
        self.c1 = c1
        self.c2 = c2
        self.k = k
        self.b = b
        self.e1 = 0.    # e1 = xd - x
        self.e2 = 0.    # e2 = e1_dot + c1 * e1
        self.e3 = 0.    # e3 = e2_dot + c2 * e2
        self.output = 0.

    def setRise(self, c1, c2, k, b):
        self.c1 = c1
        self.c2 = c2
        self.k = k
        self.b = b

    def set_e(self, e):
        e1_dot = e - self.e1
        self.e1 = e
        e2_dot = e1_dot + self.c1 * self.e1 - self.e2
        self.e2 = e1_dot + self.c1 * self.e1
        self.e3 = e2_dot + self.c2 * self.e2

        self.output += (self.k + 1) * self.e3 + self.b * np.tanh(self.e2)

    def out(self):
        return self.output

    def reset(self):
        self.__init__()
