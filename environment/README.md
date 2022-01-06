This directory storages some environments for us to do some simulation experiments.

Currently, there are two sub-directories: config and env.

# Config
This directory storages all model description files in it. A model description file is an 'xml' file that generated 
automatically by the corresponding environment.

Currently, there are only two models in it: 'Flight_Attitude_Simulator.xml' and 'Flight_Attitude_Simulator_Continuous.xml'.
Actually, they are identical except the action space. The action of the former is discrete while the latter is continuous. Apparently, the
former one is designed for value-based RL, while the latter one is designed for policy-based RL.

# Envs
This directory contains source files of the models. However, it contains more files than that of in 'config' because some environments are defined and 
described by more than one python files.

To figure out which python file is the 'exact' source file of the 'xml', we can check the name of the file, which should be identical to the corresponding
'xml' file in 'config'.