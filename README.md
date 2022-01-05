# ReinforcementLearning
Currently, this repository consists of algorithm, common, datasave, environment, and simulation.

Algorithm includes the core reinforcement learning algorithms, DQN, DDOG, for example.

Common includes common.py, which has some basic functions. Details of is can be found in common.py

Datasave saves networks trained by us, log files, and some data files.

Environment contains some physical models, which are call 'environment' in RL.
The 'config' directory contains the model description file for model, which is a '.xml' file.
THe 'envs' directory is the detailed implementation of all the physical models.

Simulation is the place where we implement our simulation experiments, which means, using different algorithms in different environments.

## A Demo
Currently, we only have a DQN controller for 'FlightAttitudeSimulator'.

'DQN' can be found in /algorithm/value_base.

'FlightAttitudeSimulator' can be found in /environment/envs/flight_attitude_simulator.py, and the model description file is '/environment/config/Flight_Attitude_Simulator.xml'

We have a trained network named 'dqn-4-flight-attitude-simulator.pkl' in 'datasave'.
However, the controller can stable the system but have steady error.

To test the network, we need to enter '/simulation/DQN_based/'DQN-4-Flight-Attitude-Simulator.py,
set:
```
TRAIN: False
RETRAIN: False
TEST: True
```
Then run this python file, you can see the result.
(Actually, the converge process is much faster than that of in the gif, probably because of something wrong with this gif file.)

![image](https://github.com/ReinforcementLearning-StudyNote/ReinforcementLearning/blob/main/datasave/video/dqn-4-flight-attitude-simulator.gif)
Wow~~~
