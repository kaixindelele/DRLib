import numpy as np
import matplotlib.pyplot as plt
import gym
import cv2
from gym.envs.robotics.fetch.drawer import FetchDrawerEnv
env = FetchDrawerEnv()


def p_control(env, obs, p_rate=0.1):
    a = env.action_space.sample()
    gg = obs['grip_goal']
    ag = obs['achieved_goal']
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
    # if np.random.random() < 0.1:
    #     action[-1] = 1.0
    # else:
    #     action[-1] = 0.0
    return action


for ep in range(20):
    ag_list = []
    obs = env.reset()
    move = False
    for i in range(50):
        if not move:
            a = p_control(env, obs=obs)
        gg2ag = np.linalg.norm(obs['grip_goal'] - obs['achieved_goal'])
        print("gg2ag:", gg2ag)
        if gg2ag < 0.03:
            a = env.action_space.sample()
            a[0] = -0.1
            a[-1] = 0
        #     move = True
        # print("a:", a)
        # a = env.action_space.sample()
        a[2] = -1.0
        obs, reward, done, info = env.step(a)
        print("ep:{}, i:{}, reward:{}, done:{}, info:{}".format(ep, i, reward, done, info))
        print("ag:", obs['achieved_goal'])
        print("gg:", obs['grip_goal'])
        ag_list.append(obs['achieved_goal'])

        # env.render()
        image_size = 2048
        img = env.render(mode='rgb_array', width=image_size, height=image_size)
        clip_value = 200
        # [上下， 左右，:]
        img = img[clip_value*2:image_size-1*clip_value, 0:image_size-2*clip_value, :]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('drawer2.png', img)

    # plt.plot(ag_list)
    # plt.pause(2)
