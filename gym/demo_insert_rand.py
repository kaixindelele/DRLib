import numpy as np
import matplotlib.pyplot as plt
import gym
from gym.envs.robotics.fetch.insert import FetchInsertEnv
import cv2
import matplotlib.pyplot as plt
env = FetchInsertEnv()


def p_control(env, obs, p_rate=0.2):
    a = env.action_space.sample()
    gg = obs['grip_goal']
    ag = obs['achieved_goal']
    ag[0] -= 0.01
    ag[1] += 0.01
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
    return action


for ep in range(20):
    obs = env.reset()
    # center = [1.35, 0.8]
    # (w1x, w1y), (w2x, w2y), (w3x, w3y), (w4x, w4y) = env.reset_hole(hole_center=center)
    #
    # env.reset_hole(center)
    for i in range(200):
        gg = obs['grip_goal']
        ag = obs['achieved_goal']
        dist = np.linalg.norm(gg[:2]-ag[:2])
        if dist > 0.01:
            a = p_control(env, obs=obs)
        else:
            a = env.action_space.sample()
        # a[0] = 0.01
        # if obs['grip_goal'][2] < 0.3:
        #     pass
        # else:
        #     a[1] = -0.2
        # a[2] = -0.2
        a *= 0.0
        print("gg:", obs['grip_goal'])
        print("ag:", obs['achieved_goal'])
        print("dg:", obs['desired_goal'])
        # env.sim.model.body_pos[env.sim.model.body_name2id('w3')] = np.array([1.45, 0.85, 0.41])
        obs, reward, done, info = env.step(a)
        print("ep:{}, i:{}, reward:{}, done:{}, info:{}".format(ep, i, reward, done, info))

        env.render()

        # image_size = 2048
        # img = env.render(mode='rgb_array', width=image_size, height=image_size)
        # clip_value = 200
        # # [上下， 左右，:]
        # img = img[clip_value*2:image_size-1*clip_value, 0:image_size-2*clip_value, :]
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('insert_rand.png', img)
