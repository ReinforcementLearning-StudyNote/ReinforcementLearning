from uav import UAV
from environment.envs.PIDControl.pid import PID
from common.common_func import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import math


if __name__ == '__main__':
	pos0 = [0, 0, 0]  # 给的都是惯性系 东北天
	angle0 = [deg2rad(45), deg2rad(0), deg2rad(0)]  # 给的都是惯性系 东北天

	quad = UAV(pos0=pos0, angle0=angle0)	# initialization of a quadrotor
	pid_x = PID(kp=0., ki=0., kd=0.)		# controller of x
	pid_y = PID(kp=0., ki=0., kd=0.)		# controller of y
	pid_z = PID(kp=0., ki=0., kd=0.)		# controller z
	pid_roll = PID(kp=0., ki=0., kd=0.)		# controller of roll along X in world
	pid_pitch = PID(kp=0., ki=0., kd=0.)	# controller of pitch along Y in world
	pid_yaw = PID(kp=0., ki=0., kd=0.)  	# controller of yaw along Y in world

	# quad.set_position_limitation2inf()		# 将无人机的位置限制设置为infinity，方便调试，无人机将始终停留在视野中间

	'''visualization'''
	fig, ax = plt.subplots(figsize=(8, 4), ncols=2, nrows=1)
	ax[0].set_xlim((quad.xmin, quad.xmax))
	ax[0].set_xlim((quad.ymin, quad.ymax))
	ax[0].set_xticks(np.arange(quad.xmin, quad.xmax, 2))
	ax[0].set_yticks(np.arange(quad.ymin, quad.ymax, 2))

	ax[1].set_ylim((quad.zmin, quad.zmax))
	ax[1].yaxis.set_major_locator(MultipleLocator(2))

	traj_xy, = ax[0].plot([], [], color='blue', linewidth=2)		# 画轨迹
	cur_pos_xy, = ax[0].plot([], [], marker='o', color='red', markersize=6)# 画当前

	traj_z, = ax[1].plot([], [], color='blue', linewidth=2)  # 画轨迹
	cur_pos_z, = ax[1].plot([], [], marker='o', color='red', markersize=6)  # 画当前

	traj_ref = [[], [], []]
	t = []
	count = 0
	'''visualization'''

	plt.ion()

	while not quad.is_episode_Terminal():
		'''1. 生成参考轨迹'''
		x_ref = 8 * math.cos(2 * math.pi * quad.time) / (1 + math.sin(2 * math.pi * quad.time) ** 2)
		y_ref = 16 * math.sin(2 * math.pi * quad.time) * math.cos(2 * math.pi * quad.time) / (1 + math.sin(2 * math.pi * quad.time) ** 2)
		z_ref = 5 * math.sin(2 * math.pi * quad.time)
		t_ref = quad.time
		quad.time += quad.dt

		if count < 200:
			traj_ref[0].append(x_ref), traj_ref[1].append(y_ref), traj_ref[2].append(z_ref), t.append(quad.time)
		else:
			traj_ref[0].pop(0), traj_ref[1].pop(0), traj_ref[2].pop(0), t.pop(0)
			traj_ref[0].append(x_ref), traj_ref[1].append(y_ref), traj_ref[2].append(z_ref), t.append(quad.time)
		count += 1

		'''2. 计算位置 PID 控制的输出，同时得到期望姿态角'''
		ex, ey, ez = x_ref - quad.x, y_ref - quad.y, z_ref - quad.z
		pid_x.set_e(ex)
		pid_y.set_e(ey)
		pid_z.set_e(ez)
		ux, uy, uz = pid_x.out(), pid_y.out(), pid_z.out()
		U1 = quad.m * math.sqrt(ux ** 2 + uy ** 2 + (uz + quad.g) ** 2)
		psi_ref = 0
		phi_ref = math.asin((ux * math.sin(psi_ref) - uy * math.cos(psi_ref)) * quad.m / U1)
		theta_ref = math.asin(ux * quad.m - U1 * math.sin(psi_ref) * math.sin(phi_ref)) / (U1 * math.cos(psi_ref) * math.cos(phi_ref))

		'''3. 计算姿态 PID 的控制输出'''

		'''4. 绘制图形'''
		traj_xy.set_data(traj_ref[0], traj_ref[1])
		cur_pos_xy.set_data([x_ref], [y_ref])

		ax[1].set_xlim((t[0], t_ref))
		traj_z.set_data(t, traj_ref[2])
		cur_pos_z.set_data([t_ref], [z_ref])
		plt.show()
		plt.pause(0.0000000001)

	plt.ioff()
