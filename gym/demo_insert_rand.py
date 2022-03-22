import numpy as np
import matplotlib.pyplot as plt
import gym
from gym.envs.robotics.fetch.insert_rand import FetchInsertRandEnv
env = FetchInsertRandEnv()


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
    obs = env.reset()
    center = [1.35, 0.8]
    (w1x, w1y), (w2x, w2y), (w3x, w3y), (w4x, w4y) = env.reset_hole(hole_center=center)

    env.reset_hole(center)
    for i in range(200):
        a = p_control(env, obs=obs)
        #
        a = env.action_space.sample()
        # a[0] = 0.01
        # if obs['grip_goal'][2] < 0.3:
        #     pass
        # else:
        #     a[1] = -0.2
        a[2] = -0.2
        print("gg:", obs['grip_goal'])
        # env.sim.model.body_pos[env.sim.model.body_name2id('w3')] = np.array([1.45, 0.85, 0.41])
        obs, reward, done, info = env.step(a)
        print("ep:{}, i:{}, reward:{}, done:{}, info:{}".format(ep, i, reward, done, info))
        env.render()
