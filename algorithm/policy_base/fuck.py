import time

import numpy as np

from common.common_cls import *
import torch.multiprocessing as mp
from multiprocessing import shared_memory

class DPPO2:
	def __init__(self):
		self.ref_buffer = np.ones(5, dtype=np.float32)
		self.share_memory = shared_memory.SharedMemory(create=True, size=self.ref_buffer.nbytes)
		self.buffer1 = np.ndarray(self.ref_buffer.shape, self.ref_buffer.dtype, self.share_memory.buf)
		self.buffer1[:] = self.ref_buffer[:]
		self.value = mp.Value('i', 1)

	def send(self):
		try:
			while True:
				for i in range(5):
					self.buffer1[i] += 1
				time.sleep(0.1)
				print('send:', self.buffer1)
		except:
			self.clean()

	def clean(self):
		self.share_memory.close()
		self.share_memory.unlink()
		print('共享内存清理完毕')


def receive(sh:shared_memory.SharedMemory):
	# n_sh = shared_memory.SharedMemory(name=sh.name)
	h = np.ndarray((5,), np.float32, sh.buf)
	try:
		while True:
			print('receive', h)
			time.sleep(0.2)
	except:
		# n_sh.close()
		print('退出')


if __name__ == '__main__':
	mp.set_start_method('spawn', force=True)
	agent = DPPO2()
	p1 = mp.Process(target=agent.send, args=())
	p2 = mp.Process(target=receive, args=(agent.share_memory,))
	p1.start()
	p2.start()
	p1.join()
	p2.join()
