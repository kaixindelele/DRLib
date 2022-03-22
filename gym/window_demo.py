import numpy as np
import matplotlib.pyplot as plt
import gym
from gym.envs.robotics.fetch.window_open import FetchWindowOpenEnv
env = FetchWindowOpenEnv()


def p_control(env, obs, p_rate=0.1):
    a = env.action_space.sample()
    gg = obs['grip_goal']
    ag = obs['achieved_goal']
    ag[1] -= 0.05
    ag[2] -= 0.02
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
    if np.random.random() < 0.5:
        action[-1] = 1.0
    else:
        action[-1] = 0.0
    return action


for ep in range(20):
    ag_list = []
    obs = env.reset()
    move = False
    for i in range(100):
        # if not move:
        #     a = p_control(env, obs=obs)
        # gg2ag = np.linalg.norm(obs['grip_goal'] - obs['achieved_goal'])
        # print("gg2ag:", gg2ag)
        # if gg2ag < 0.02:
        #     a[1] = 0.5
        #     move = True
        # print("a:", a)
        a = env.action_space.sample()
        a[0] = 0.1
        obs, reward, done, info = env.step(a)
        print("ep:{}, i:{}, reward:{}, done:{}, info:{}".format(ep, i, reward, done, info))
        print("ag:", obs['achieved_goal'])
        ag_list.append(obs['achieved_goal'])

        env.render()
    plt.plot(ag_list)
    plt.show()
