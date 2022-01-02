import random
from environment.envs.flight_attitude_simulator import Flight_Attitude_Simulator


'''Flight Attitude Simulator Test'''
def test_flight_attitude_simulator():
    env = Flight_Attitude_Simulator(initTheta=-60.0, setTheta=0., save_cfg=True)
    test_num = 1
    for _ in range(test_num):
        env.reset_random()
        while not env.is_terminal:
            env.show_dynamic_image(isWait=False)
            # action = random.choice(env.action_space)
            action = [0.0]
            env.current_state, env.current_action[0], env.reward, env.next_state, env.is_terminal = env.step_update(action=action)
    # env.saveData(is2file=True, filepath='../../datasave/')


test_flight_attitude_simulator()
'''Flight Attitude Simulator Test'''
