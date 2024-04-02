import numpy as np
import matplotlib.pyplot as plt
import gym
import time
from gym.envs.robotics.fetch.tstack import FetchThreeStackEnv
env = FetchThreeStackEnv()


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


def obs2state(obs):
    cur_mask = [1, 1, 1, 1]
    output_keys = ['ag0', 'dg0', 'dg1', 'dg2']
    s_goal = np.concatenate([obs[key] * cur_mask[index] for index, key in enumerate(output_keys)])

    s = np.concatenate([obs['observation'], s_goal])
    return s


state_list = []
for ep in range(2000):
    st = time.time()
    obs = env.reset(success_list=[4, 1,])
    for i in range(50):
        if np.random.random() < 0.5:
            a = p_control(env, obs=obs)
        else:
            a = env.action_space.sample()
        # print("gg:", obs['grip_goal'])

        obs, reward, done, info = env.step(a)
        # print("ep:{}, i:{}, reward:{}, done:{}, info:{}".format(ep, i, reward, done, info))
        state = obs2state(obs)
        # print("state:", state)
        state_list.append(state)
        env.render()
    print('ep_time:', time.time() - st)

# state_array = np.array(state_list)
# state_std = np.std(state_array, axis=0)
# state_mean = np.mean(state_array, axis=0)
# np.set_printoptions(precision=4, suppress=True)
# print("state_std:", state_std)
# print('[', end='')
# for v in state_std:
#     print(np.round(v, 2), end='')
#     print(', ', end='')
# # [print(v, ', ', end='') for v in state_std]
# print(']')
#
# print("state_mean:", state_mean)
#
# print('[', end='')
# for v in state_mean:
#     print(np.round(v, 2), end='')
#     print(', ', end='')
# print(']')
