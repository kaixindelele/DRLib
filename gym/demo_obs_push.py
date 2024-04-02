import numpy as np
import matplotlib.pyplot as plt
import gym
import cv2
from gym.envs.robotics.fetch.obs_push import FetchObsPushEnv
env = FetchObsPushEnv()


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
    for i in range(20):
        a = p_control(env, obs=obs)
        a[-1] = 0.0
        # a = env.action_space.sample()
        # a[0] = 0.01
        # if obs['grip_goal'][2] < 0.3:
        #     pass
        # else:
        #     a[1] = -0.2
        # a[2] = -0.2
        print("gg:", obs['grip_goal'])
        a *= 0
        obs, reward, done, info = env.step(a)
        print("ep:{}, i:{}, reward:{}, done:{}, info:{}".format(ep, i, reward, done, info))
        # env.render()
        image_size = 2048
        img = env.render(mode='rgb_array', width=image_size, height=image_size)
        clip_value = 200
        # [上下， 左右，:]
        img = img[clip_value*2:image_size-1*clip_value, 0:image_size-2*clip_value, :]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('obstacle_push.png', img)
