import copy
import time

import numpy as np
import torch

from common.common_func import *
import torch.multiprocessing as mp
from multiprocessing import shared_memory
import sys, os


class Write(mp.Process):
    def __init__(self):
        super(Write, self).__init__()
        ref = np.ones((2, 10), dtype=np.float32)
        bite = np.zeros(1, dtype=int)
        self.share_memory = shared_memory.SharedMemory(create=True, size=ref.nbytes)
        self.bool_memory = shared_memory.SharedMemory(create=True, size=bite.nbytes)
        self.buffer = np.ndarray(ref.shape, ref.dtype, self.share_memory.buf)
        self.value = np.ndarray(bite.shape, bite.dtype, self.bool_memory.buf)
        self.value[0] = 1

    def run(self) -> None:
        try:
            i = 0
            while i < 10:
                if self.value[0] == 1:
                    temp = i * torch.ones((2,10), dtype=torch.float32)
                    self.buffer[:] = temp.numpy()[:]
                    i += 1
                    # time.sleep(0.5)
                    self.value[0] = 0
                else:
                    pass
            '''说明 i > 10'''
            # while self.value[0] != 2:
            #     pass
                # print('waiting')
            print('write end')
            self.share_memory.close()
            self.share_memory.unlink()
        except:
            self.share_memory.close()
            self.share_memory.unlink()


class Read(mp.Process):
    def __init__(self,
                 share_memory:shared_memory.SharedMemory.buf,
                 share_bool:shared_memory.SharedMemory.buf,
                 ref_buffer:np.ndarray,
                 ref_bool: np.ndarray):
        super(Read, self).__init__()
        self.buffer = np.ndarray(ref_buffer.shape, ref_buffer.dtype, share_memory)
        self.value = np.ndarray(ref_bool.shape, ref_bool.dtype, share_bool)

    def run(self) -> None:
        try:
            count = 0
            while True:
                if self.value[0] == 0:
                    print(self.buffer)
                    self.value[0] = 1
                    count += 1
                    print('read', count)
                    if count == 10:
                        self.value[0] = 2
                        break
                else:
                    pass
            print('read end')
        except:
            pass


os.environ["OMP_NUM_THREADS"] = "1"


if __name__ == '__main__':
    a = []
    for _ in range(1):
        a.append(np.ones((2,2)))
    b = np.vstack(a)
    print(a)
    print(b)
    # mp.set_start_method('fork', force=True)
    # write = Write()
    # read = Read(write.share_memory.buf, write.bool_memory.buf, write.buffer, write.value)
    # # mp.set_start_method(mp.get_start_method(), force=True)
    #
    # write.start()
    # read.start()
    #
    # write.join()
    # read.join()
