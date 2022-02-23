# Installation
This platform can be implemented ob both windows and ubuntu since it is all established by Python. Currently, there is no ROS, Gazebo or PX4 related packages.
## Python version
Python 3.6, 3.7 and 3.9 have been tested. (There seems to be a bug in OpenCV-python3.8, which has something to do with visualization. But I forgot, sorry...)
## Installation
Pre-installed: Anaconda3, any version that has a default python3.6, 3.7 or 3.9 is fine.
```commandline
pip install opencv-python
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
The version of PyTorch depends on the device you have. You can choose CPU only or a specified CUDA version different from mine.
# ReinforcementLearning
Currently, this repository consists of algorithm, common, datasave, environment, and simulation.

Algorithm includes the core reinforcement learning algorithms, DQN, DDPG, TD3, for example.

Common includes common.py that has some basic functions, cross product of two vectors, vector transformation, rotation, etc. Details of them can be found in common.py

Datasave saves networks trained by RL algorithms, and some data files.

Environment contains some physical models, which are called 'environment' in RL.
The 'config' directory contains the model description files for all environments, which are '**.xml' files.
THe 'envs' directory is the detailed implementation of all the physical environments.

Simulation is the place where we implement our simulation experiments, which means, using different algorithms in different environments.

## Demos
Currently, we only six well-trained agents.

1. A DDPG controller for a flight attitude simulator.
2. A DDPG trajectory planner for a bidirectional UGV.
3. A DDPG trajectory planner for a forward-only UGV.
4. A DDPG trajectory planner which can avoid obstacles for a forward-only UGV.
5. A DQN controller for a flight attitude simulator.
6. A TD3 trajectory planner which can avoid obstacles for a forward-only UGV.

## Run the scripts
All runnable scripts are in './simulation/'.
### A DQN controller for a flight attitude simulator.
```commandline
cd simulation/DQN_based/
python DQN-4-Flight-Attitude-Simulator.py
```
The result should be similar to the following.

<div align=center>
    <img src="https://github.com/ReinforcementLearning-StudyNote/ReinforcementLearning/blob/main/datasave/video/gif/dqn-4-flight-attitude-simulator.gif" width="400px">
</div>

In 'DQN-4-Flight-Attitude-Simulator.py', set:
```commandline
    TRAIN = False
    RETRAIN = False
    TEST = not TRAIN
```
### A DDPG trajectory planner which can avoid obstacles for a forward-only UGV.
```commandline
cd simulation/PG_based/
python DDPG-4-UGV-Forward-Obstacle.py
```
The result should be similar to the following.

In 'DDPG-4-UGV-Forward-Obstacle.py', set:
```commandline
    TRAIN = False
    RETRAIN = False
    TEST = not TRAIN
```
<div align=center>
    <img src="https://github.com/ReinforcementLearning-StudyNote/ReinforcementLearning/blob/main/datasave/video/gif/DDPG-4-UGV-Obstacle1.gif" width="400px"><img src="https://github.com/ReinforcementLearning-StudyNote/ReinforcementLearning/blob/main/datasave/video/gif/DDPG-4-UGV-Obstacle2.gif" width="400px">
</div>

### A DDPG trajectory planner in an empty world for a forward-only UGV.
```commandline
cd simulation/PG_based/
python DDPG-4-UGV-Forward.py
```
The result should be similar to the following.

In 'DDPG-4-UGV-Forward.py', set:
```commandline
    TRAIN = False
    RETRAIN = False
    TEST = not TRAIN
```
<div align=center>
    <img src="https://github.com/ReinforcementLearning-StudyNote/ReinforcementLearning/blob/main/datasave/video/gif/DDPG-4-UGV-Forward1.gif" width="400px"><img src="https://github.com/ReinforcementLearning-StudyNote/ReinforcementLearning/blob/main/datasave/video/gif/DDPG-4-UGV-Forward2.gif" width="400px">
</div>