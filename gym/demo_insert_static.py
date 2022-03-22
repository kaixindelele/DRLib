import numpy as np
import matplotlib.pyplot as plt
import gym
from gym.envs.robotics.fetch.insert import FetchInsertEnv
env = FetchInsertEnv()


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
        # if axis == 0:
        #     a[axis] = -p_rate
        # else:
        #     a[axis] = p_rate
    action = a
    return action


for ep in range(20):
    obs = env.reset()
    for i in range(200):
        # a = p_control(env, obs=obs)
        #
        a = env.action_space.sample()
        a[0] = 0.01
        if obs['grip_goal'][2] < 0.3:
            pass
        else:
            a[1] = -0.2
        a[2] = -0.2
        print("gg:", obs['grip_goal'])

        obs, reward, done, info = env.step(a)
        print("ep:{}, i:{}, reward:{}, done:{}, info:{}".format(ep, i, reward, done, info))
        env.render()
