import numpy as np
import matplotlib.pyplot as plt
import gym
from gym.envs.robotics.fetch.drawer_open import FetchDrawerOpenEnv
from gym.envs.robotics.fetch.drawer_horiz_open import FetchHorizonDrawerOpenEnv
from gym.envs.robotics.fetch.drawer_box import FetchDrawerBoxEnv
env = FetchDrawerBoxEnv()
# env = FetchHorizonDrawerOpenEnv()


def p_control(env, obs, p_rate=0.99):
    a = env.action_space.sample()
    gg = obs['grip_goal']
    ag = obs['ag0']
    ag[1] -= 0.01
    # ag[0] += 0.05
    error = ag - gg
    for axis, value in enumerate(error):
        if abs(value) > 0.01:
            if value > 0:
                a[axis] = p_rate
            else:
                a[axis] = -p_rate
        else:
            a[axis] = 0
    action = a
    # action = np.zeros(4)
    # if np.random.random() < 0.1:
    #     action[-1] = 1.0
    # else:
    #     action[-1] = 0.0
    return action


env.task = 'in2out'
# env.task = 'out2in'
for ep in range(20):
    ag_list = []
    obs = env.reset()
    move = False
    for i in range(100):
        if not move:
            a = p_control(env, obs=obs)
        gg2ag = np.linalg.norm(obs['grip_goal'] - obs['ag0'])
        print("gg2ag:", gg2ag)
        if gg2ag < 0.05:
            # a = env.action_space.sample()
            a[0] = -0.1
            a[-1] = -1.0
        #     move = True
        # print("a:", a)
        # a = env.action_space.sample()
        # a[2] = -1.0
        # if i > 60:
        #     a[-1] = 1
        #     a[2] = 1
        obs, reward, done, info = env.step(a)
        print("ep:{}, i:{}, reward:{}, done:{}, info:{}".format(ep, i, reward, done, info))
        print("ag:", obs['achieved_goal'])
        print("ag1:", obs['ag1'])
        print("gg:", obs['grip_goal'])
        ag_list.append(obs['achieved_goal'])

        env.render()
    # plt.plot(ag_list)
    # plt.pause(2)
