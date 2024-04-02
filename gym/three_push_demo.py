import numpy as np
import matplotlib.pyplot as plt
import gym
import time
from gym.envs.robotics.fetch.tpush import FetchThreePushEnv
env = FetchThreePushEnv()


def p_control(env, obs, p_rate=0.2):
    a = env.action_space.sample()
    gg = obs['grip_goal']
    ag = obs['achieved_goal']
    error = ag - gg
    for axis, value in enumerate(error):
        if abs(value) > 0.02:
            if value > 0:
                a[axis] = p_rate
            else:
                a[axis] = -p_rate
        else:
            a[axis] = 0
    action = a
    return action


for ep in range(20):
    st = time.time()
    obs = env.reset()
    for i in range(50):
        # a = p_control(env, obs=obs)
        # a[2] = 0.0
        a = np.random.random(4)
        a[2] = 1.0

        print("gg:", obs['grip_goal'])

        obs, reward, done, info = env.step(a)
        print("obs:", obs)
        print("ep:{}, i:{}, reward:{}, done:{}, info:{}".format(ep, i, reward, done, info))
        env.render()
    # print('ep_time:', time.time() - st)
