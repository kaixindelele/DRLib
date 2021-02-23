"""
1. 换成了per_buffer的类，在td3的类中，变动内容有：
    1.增加计算abs_error的句柄：
    self.abs_errors = tf.abs(min_q_targ - self.q1)
    2.更新q值的时候，增加重要性采样的系数
    self.q_loss = self.ISWeights * (q1_loss + q2_loss)
    3.修改存储transition，因为per_buffer里面，存的格式是列表，不需要拆解transition。
    self.replay_buffer.store(transition)
    4.修改更新参数的函数，对buffer的采样，数据格式也不一样：
    tree_idx, batch_memory, ISWeights = self.replay_buffer.sample(batch_size=batch_size)
    但这个其实改的可以很少。
2. 调用TD3_per和普通TD3没有任何区别。
3. 做了测试对比，在HalfCheetah-v2任务中，TD3_per每个回合更新时间是26秒，TD3的只要4秒，实在是浪费时间。这里面也可能是因为我的cpu资源被其他进程占满了。性能这块的话，其实影响不大。

"""

import numpy as np
import tensorflow as tf
import gym
import time
import numpy as np
import tensorflow as tf
import gym
import os
import time
import sys
import copy

sys.path.append("../")

from rl_algorithms.td3_sp import core
from rl_algorithms.td3_sp.core import get_vars
# try:
#     from her_utils import core
#     from her_utils.core import get_vars
# except Exception as e:
#     print("import td3 e:", e)
#     from td3_sp import core
#     from td3_sp.core import get_vars


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """
    def __init__(self, 
                 obs_dim, 
                 act_dim, 
                 size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


class TD3:
    def __init__(self,
                 a_dim, obs_dim, a_bound,
                 mlp_actor_critic=core.mlp_actor_critic,
                 ac_kwargs=dict(),
                 seed=0,
                 replay_size=int(1e6), gamma=0.9,
                 polyak=0.99,
                 pi_lr=1e-3, q_lr=1e-3,
                 act_noise=0.1, target_noise=0.2,
                 noise_clip=0.5, policy_delay=2,
                 sess_opt=None,
                 sess = None,
                 batch_size=256,
                 buffer=None,
                 per_flag=True,
                 her_flag=True,
                 goal_selection_strategy="future",
                 n_sampled_goal=4,
                 ):
        self.learn_step = 0
        self.obs_dim = obs_dim
        self.act_dim = a_dim
        self.act_limit = a_bound
        self.policy_delay = policy_delay
        self.action_noise = act_noise

        # Share information about action space with policy architecture
        ac_kwargs['action_space'] = a_bound

        # Inputs to computation graph
        self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
        self.x_ph, self.a_ph, self.x2_ph, self.r_ph, self.d_ph = core.placeholders(obs_dim, a_dim, obs_dim, None, None)
        self.actor_lr = tf.placeholder(tf.float32, shape=[], name='actor_lr')
        self.critic_lr = tf.placeholder(tf.float32, shape=[], name='critic_lr')

        # Main outputs from computation graph
        with tf.variable_scope('main'):
            self.pi, self.q1, self.q2, self.q1_pi = mlp_actor_critic(self.x_ph, self.a_ph, **ac_kwargs)

        # Target policy network
        with tf.variable_scope('target'):
            pi_targ, _, _, _ = mlp_actor_critic(self.x2_ph, self.a_ph, **ac_kwargs)

        # Target Q networks
        with tf.variable_scope('target', reuse=True):
            # Target policy smoothing, by adding clipped noise to target actions
            epsilon = tf.random_normal(tf.shape(pi_targ), stddev=target_noise)
            epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
            a2 = pi_targ + epsilon
            a2 = tf.clip_by_value(a2, -self.act_limit, self.act_limit)

            # Target Q-values, using action from target policy
            _, q1_targ, q2_targ, _ = mlp_actor_critic(self.x2_ph, a2, **ac_kwargs)

        self.per_flag = per_flag
        self.her_flag = her_flag
        self.goal_selection_strategy = goal_selection_strategy
        self.n_sampled_goal = n_sampled_goal
        # Experience buffer

        if self.per_flag:
            try:
                from rl_algorithms.memory.sp_per_memory import ReplayBuffer
            except:
                from memory.sp_per_memory import ReplayBuffer
        else:
            try:
                from rl_algorithms.memory.sp_memory import ReplayBuffer
            except:
                from memory.sp_memory import ReplayBuffer
        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=self.act_dim, size=replay_size)

        # Count variables
        var_counts = tuple(core.count_vars(scope)
                           for scope in ['main/pi',
                                         'main/q1',
                                         'main/q2',
                                         'main'])
        print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d, \t total: %d\n' % var_counts)
            
        # Bellman backup for Q functions, using Clipped Double-Q targets
        min_q_targ = tf.minimum(q1_targ, q2_targ)
        # backup = tf.stop_gradient(self.r_ph + gamma * (1 - self.d_ph) * min_q_targ)
        backup = self.r_ph + gamma * min_q_targ
        
        # TD3 losses
        self.pi_loss = -tf.reduce_mean(self.q1_pi)
        q1_loss = tf.reduce_mean((self.q1 - backup) ** 2)
        q2_loss = tf.reduce_mean((self.q2 - backup) ** 2)
        
        if self.per_flag:
            # 也许可以选q2,但是一般来说q1和q2值相差不大。之前写错了，不是q_targ，莫烦的q_targ是因为他的q_t写成了q_
            self.abs_errors = tf.abs(backup - self.q1)
            # 是不是这么乘的，我也迷惑，等下测试一下，看看效果有没有提升。
            self.q_loss = self.ISWeights * (q1_loss + q2_loss)
        else:
            # 正常的！
            self.q_loss = q1_loss + q2_loss
        
        # Separate train ops for pi, q
        pi_optimizer = tf.train.AdamOptimizer(learning_rate=self.actor_lr)
        q_optimizer = tf.train.AdamOptimizer(learning_rate=self.critic_lr)
        self.train_pi_op = pi_optimizer.minimize(self.pi_loss,
                                                 var_list=get_vars('main/pi'))
        # 这里的参数,怎么是总的q?
        # 难道这里的字符串只需要匹配就好了?
        self.train_q_op = q_optimizer.minimize(self.q_loss,
                                               var_list=get_vars('main/q'))

        # Polyak averaging for target variables
        self.target_update = tf.group([tf.assign(v_targ, polyak * v_targ + (1 - polyak) * v_main)
                                       for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        # Initializing targets to match main variables
        target_init = tf.group([tf.assign(v_targ, v_main)
                                for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        if sess_opt:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=sess_opt)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        else:
            self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(target_init)

    def get_action(self, s, noise_scale=0):
        a = self.sess.run(self.pi,
                          feed_dict={self.x_ph: s.reshape(1, -1),
                                              })[0]
        a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)

    def get_q(self, s, a):
        Q = self.sess.run([self.q1, self.q2],
                           feed_dict={self.x_ph: s.reshape(1, -1),
                           self.a_ph: a.reshape(1, -1),
                           })
        q1, q2 = Q[0][0], Q[1][0]
        return q1, q2

    def store_transition(self, transition):
        if self.per_flag:
            self.replay_buffer.store(transition)
        else:
            (s, a, r, s_, done) = transition
            self.replay_buffer.store(s, a, r, s_, done)

    #### HER utils
    def save_episode(self, episode_trans, reward_func):
        # 存的时候就替换目标，其实不合适，数据变化会受限。
        for transition_idx, transition in enumerate(episode_trans):
            # We cannot sample a goal from the future in the last step of an episode
            if (transition_idx == len(episode_trans) - 1 and
                    self.goal_selection_strategy == "future"):
                break
            obs, action, reward, next_obs, done, info = copy.deepcopy(transition)
            obs_arr, next_obs_arr = map(self.convert_dict_to_array,
                                        (obs, next_obs))

            self.store_transition(transition=(obs_arr, action, reward, next_obs_arr, done))
            sampled_goals = self._sample_achieved_goals(episode_trans, transition_idx,
                                                        n_sampled_goal=self.n_sampled_goal)
            # For each sampled goals, store a new transition
            for goal in sampled_goals:
                # Copy transition to avoid modifying the original one
                # 默认obs是字典格式
                obs, action, reward, next_obs, done, info = copy.deepcopy(transition)
                obs['desired_goal'] = goal
                next_obs['desired_goal'] = goal
                # Update the reward according to the new desired goal
                reward = reward_func(next_obs['achieved_goal'],
                                     goal, info)
                # Can we use achieved_goal == desired_goal?
                done = False
                # Transform back to ndarrays
                # map(func, (param1, param2))
                obs_arr, next_obs_arr = map(self.convert_dict_to_array, (obs, next_obs))
                # Add artificial transition to the replay buffer
                self.store_transition(transition=(obs_arr, action, reward, next_obs_arr, done))

    def _sample_achieved_goal(self, episode_transitions, transition_idx):
        """
        Sample an achieved goal according to the sampling strategy.
        :param episode_transitions: ([tuple]) a list of all the transitions in the current episode
        :param transition_idx: (int) the transition to start sampling from
        :return: (np.ndarray) an achieved goal
        """
        if self.goal_selection_strategy == "future":
            # Sample a goal that was observed in the same episode after the current step
            selected_idx = np.random.choice(np.arange(transition_idx + 1, len(episode_transitions)))
            selected_transition = episode_transitions[selected_idx]
        elif self.goal_selection_strategy == "final":
            # Choose the goal achieved at the end of the episode
            selected_transition = episode_transitions[-1]
        else:
            raise ValueError("Invalid goal selection strategy,"
                             )
        ag = selected_transition[0]['achieved_goal']
        return ag

    def _sample_achieved_goals(self, episode_transitions, transition_idx, n_sampled_goal=4):
        """
        Sample a batch of achieved goals according to the sampling strategy.
        :param episode_transitions: ([tuple]) list of the transitions in the current episode
        :param transition_idx: (int) the transition to start sampling from
        :return: (np.ndarray) an achieved goal
        返回k个新目标元组
        """
        return [
            self._sample_achieved_goal(episode_transitions, transition_idx)
            for _ in range(n_sampled_goal)
        ]

    def convert_dict_to_array(self, obs_dict):
        obs_array = np.concatenate([obs_dict[key] for key, value in obs_dict.items()])
        return obs_array


    def test_agent(self, env, max_ep_len=1000, n=5):
        ep_reward_list = []
        for j in range(n):
            s = env.reset()
            ep_reward = 0
            for i in range(max_ep_len):
                # Take deterministic actions at test time (noise_scale=0)
                s, r, d, _ = env.step(self.get_action(s))
                ep_reward += r
            ep_reward_list.append(ep_reward)
        mean_ep_reward = np.mean(np.array(ep_reward_list))
        return mean_ep_reward

    def learn(self, batch_size=100,
              actor_lr_input=0.001,
              critic_lr_input=0.001,
              ):

        if self.per_flag:
            tree_idx, batch_memory, ISWeights = self.replay_buffer.sample(batch_size=batch_size)
            batch_states, batch_actions, batch_rewards, batch_states_, batch_dones = [], [], [], [], []
            for i in range(batch_size):
                batch_states.append(batch_memory[i][0])
                batch_actions.append(batch_memory[i][1])
                batch_rewards.append(batch_memory[i][2])
                batch_states_.append(batch_memory[i][3])
                batch_dones.append(batch_memory[i][4])

            feed_dict = {self.x_ph: np.array(batch_states),
                         self.x2_ph: np.array(batch_states_),
                         self.a_ph: np.array(batch_actions),
                         self.r_ph: np.array(batch_rewards),
                         self.d_ph: np.array(batch_dones),
                         self.actor_lr: actor_lr_input,
                         self.critic_lr: critic_lr_input,
                         self.ISWeights: ISWeights
                         }

            q_step_ops = [self.q_loss, self.q1,
                          self.q2, self.train_q_op,
                          self.abs_errors,
                          ]
            q_outs = self.sess.run(q_step_ops, feed_dict)
            q_loss, q1, q2, train_q_op, abs_errors = q_outs
            if self.learn_step % self.policy_delay == 0:
                # Delayed policy update
                outs = self.sess.run([self.pi_loss,
                                      self.train_pi_op,
                                      self.target_update],
                                     feed_dict)

            self.replay_buffer.batch_update(tree_idx,
                                            abs_errors)  # update priority
            self.learn_step += 1
            return q_outs
        else:
            batch = self.replay_buffer.sample_batch(batch_size)
            feed_dict = {self.x_ph: batch['obs1'],
                         self.x2_ph: batch['obs2'],
                         self.a_ph: batch['acts'],
                         self.r_ph: batch['rews'],
                         self.d_ph: batch['done'],
                         self.actor_lr: actor_lr_input,
                         self.critic_lr: critic_lr_input,
                         }
            q_step_ops = [self.q_loss, self.q1, self.q2, self.train_q_op]
            outs = self.sess.run(q_step_ops, feed_dict)
            if self.learn_step % self.policy_delay == 0:
                # Delayed policy update
                outs = self.sess.run([self.pi_loss,
                                      self.train_pi_op,
                                      self.target_update],
                                     feed_dict)
            self.learn_step += 1
            return outs

    def load_step_network(self, saver, load_path):
        checkpoint = tf.train.get_checkpoint_state(load_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(self.sess, tf.train.latest_checkpoint(load_path))
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
            self.learn_step = int(checkpoint.model_checkpoint_path.split('-')[-1])
        else:
            print("Could not find old network weights")

    def save_step_network(self, time_step, saver, save_path):
        saver.save(self.sess, save_path + 'network', global_step=time_step,
                   write_meta_graph=False)

    def load_simple_network(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(path))
        print("restore model successful")

    def save_simple_network(self, save_path):
        saver = tf.train.Saver()
        saver.save(self.sess, save_path=save_path + "/params", write_meta_graph=False)


if __name__ == '__main__':
    import argparse

    random_seed = int(time.time() * 1000 % 1000)
    random_seed = 184
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='FetchPush-v1')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=random_seed)
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default='td3_class')
    args = parser.parse_args()

    env = gym.make(args.env)
    env = env.unwrapped
    env.seed(args.seed)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high[0]

    net = TD3(a_dim, s_dim, a_bound,
              sess_opt=0.1,
              per_flag=True,
              her_flag=True
              )
    ep_reward_list = []
    test_ep_reward_list = []

    for i in range(args.epochs):
        s = env.reset()
        ep_reward = 0
        st = time.time()
        for j in range(args.max_steps):

            # Add exploration noise
            if i < 10:
                a = np.random.rand(a_dim) * a_bound
            else:
                # a = net.choose_action(s)
                a = net.get_action(s, 0.1)
            # a = noise.add_noise(a)

            a = np.clip(a, -a_bound, a_bound)

            s_, r, done, info = env.step(a)
            done = False if j == args.max_steps - 1 else done

            net.store_transition((s, a, r, s_, done))

            s = s_
            ep_reward += r
            if j == args.max_steps - 1:
                up_st = time.time()
                for _ in range(args.max_steps):
                    net.learn()

                ep_update_time = time.time() - up_st

                ep_reward_list.append(ep_reward)
                print('Episode:', i, ' Reward: %i' % int(ep_reward),
                      # 'Explore: %.2f' % var,
                      "learn step:", net.learn_step,
                      "ep_time:", np.round(time.time()-st, 3),
                      "up_time:", np.round(ep_update_time, 3),
                      )
                # if ep_reward > -300:RENDER = True

                # 增加测试部分!
                if i % 20 == 0:
                    test_ep_reward = net.test_agent(env=env, n=5)
                    test_ep_reward_list.append(test_ep_reward)
                    print("-" * 20)
                    print('Episode:', i, ' Reward: %i' % int(ep_reward),
                          'Test Reward: %i' % int(test_ep_reward),
                          )
                    print("-" * 20)

                break

    import matplotlib.pyplot as plt

    plt.plot(ep_reward_list)
    img_name = str(args.exp_name + "_" + args.env + "_epochs" +
                   str(args.epochs) +
                   "_seed" + str(args.seed))
    plt.title(img_name + "_train")
    plt.savefig(img_name + ".png")
    plt.show()
    plt.close()

    plt.plot(test_ep_reward_list)
    plt.title(img_name + "_test")
    plt.savefig(img_name + ".png")
    plt.show()

