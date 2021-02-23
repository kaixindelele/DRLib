import numpy as np
import tensorflow as tf
import gym
import os
import time
import sys

sys.path.append("../")
try:
    from rl_algorithms.sac_auto import core
    from rl_algorithms.sac_auto.core import get_vars
except:
    from sac_auto import core
    from sac_auto.core import get_vars


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, act_dim, size):
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


class SAC:
    def __init__(self,
                 a_dim, obs_dim, a_bound,
                 mlp_actor_critic=core.mlp_actor_critic,
                 ac_kwargs=dict(), seed=0,
                 replay_size=int(1e6), gamma=0.99,
                 polyak=0.995, alpha="auto",
                #  pi_lr=1e-4, q_lr=1e-4,
                #  batch_size=100,
                #  act_noise=0.1, target_noise=0.2, noise_clip=0.5, 
                #  policy_delay=2,
                 sess_opt=0.1,
                 ):

        self.learn_step = 0

        self.obs_dim = obs_dim
        self.act_dim = a_dim
        self.act_limit = a_bound
        self.policy_delay = policy_delay
        # self.action_noise = act_noise

        # Share information about action space with policy architecture
        ac_kwargs['action_space'] = a_bound

        # Inputs to computation graph
        self.x_ph, self.a_ph, self.x2_ph, self.r_ph, self.d_ph = core.placeholders(obs_dim, a_dim, obs_dim, None, None)
        self.actor_lr = tf.placeholder(tf.float32, shape=[], name='actor_lr')
        self.critic_lr = tf.placeholder(tf.float32, shape=[], name='critic_lr')

        # Main outputs from computation graph
        with tf.variable_scope('main'):
            self.mu, self.pi, logp_pi, logp_pi2, q1, q2, q1_pi, q2_pi, = mlp_actor_critic(self.x_ph,
                                                                                          self.x2_ph,
                                                                                          self.a_ph,
                                                                                          **ac_kwargs)

        # Target value network
        with tf.variable_scope('target'):
            _, _, logp_pi_, _, _, _, q1_pi_, q2_pi_ = mlp_actor_critic(self.x2_ph,
                                                                       self.x2_ph,
                                                                       self.a_ph,
                                                                       **ac_kwargs)

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim,
                                          act_dim=self.act_dim,
                                          size=replay_size)

        # Count variables
        var_counts = tuple(core.count_vars(scope) for scope in
                           ['main/pi', 'main/q1', 'main/q2', 'main/v', 'main'])
        print(('\nNumber of parameters: \t pi: %d, \t' + \
               'q1: %d, \t q2: %d, \t v: %d, \t total: %d\n') % var_counts)
        # 重新修改下面这段!
        target_entropy = (-np.prod(a_dim))

        log_alpha = tf.get_variable('log_alpha', dtype=tf.float32, initializer=0.0)
        alpha = tf.exp(log_alpha)

        alpha_loss = tf.reduce_mean(-log_alpha * tf.stop_gradient(logp_pi + target_entropy))

        alpha_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4,
                                                 name='alpha_optimizer')
        train_alpha_op = alpha_optimizer.minimize(loss=alpha_loss, var_list=[log_alpha])

        # Min Double-Q:
        min_q_pi = tf.minimum(q1_pi_, q2_pi_)

        # Targets for Q and V regression
        v_backup = tf.stop_gradient(min_q_pi - alpha * logp_pi2)
        q_backup = self.r_ph + gamma * (1 - self.d_ph) * v_backup

        # Soft actor-critic losses
        pi_loss = tf.reduce_mean(alpha * logp_pi - q1_pi)
        q1_loss = 0.5 * tf.reduce_mean((q_backup - q1) ** 2)
        q2_loss = 0.5 * tf.reduce_mean((q_backup - q2) ** 2)
        value_loss = q1_loss + q2_loss

        # Policy train op
        # (has to be separate from value train op, because q1_pi appears in pi_loss)
        pi_optimizer = tf.train.AdamOptimizer(learning_rate=self.actor_lr)
        train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))

        # Value train op
        # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
        value_optimizer = tf.train.AdamOptimizer(learning_rate=self.critic_lr)
        value_params = get_vars('main/q')
        with tf.control_dependencies([train_pi_op]):
            train_value_op = value_optimizer.minimize(value_loss,
                                                      var_list=value_params)

        # Polyak averaging for target variables
        # (control flow because sess.run otherwise evaluates in nondeterministic order)
        with tf.control_dependencies([train_value_op]):
            target_update = tf.group([tf.assign(v_targ, polyak * v_targ + (1 - polyak) * v_main)
                                      for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        # All ops to call during one training step
        self.step_ops = [pi_loss,
                         q1_loss, q2_loss,
                         q1, q2,
                         logp_pi, alpha,
                         train_pi_op,
                         train_value_op,
                         target_update,
                         train_alpha_op]

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
        if not noise_scale:
            act_op = self.mu
        else:
            act_op = self.pi
        a = self.sess.run(act_op,
                          feed_dict={self.x_ph: s.reshape(1, -1)})[0]
        return np.clip(a, -self.act_limit, self.act_limit)

    def store_transition(self, transition):
        (s, a, r, s_, done) = transition
        self.replay_buffer.store(s, a, r, s_, done)

    def test_agent(self, env, max_ep_len=200, n=5, logger=None):
        ep_reward_list = []
        for j in range(n):
            s = env.reset()
            ep_reward = 0
            for i in range(max_ep_len):
                # Take deterministic actions at test time (noise_scale=0)                
                a = self.get_action(s)
                s, r, d, _ = env.step(a)                
                ep_reward += r
            ep_reward_list.append(ep_reward)
        mean_ep_reward = np.mean(np.array(ep_reward_list))
        if logger:
            logger.store(TestEpRet=mean_ep_reward)
        if logger:
            return mean_ep_reward, logger
        else:
            return mean_ep_reward

    def learn(self, batch_size=100,
              actor_lr_input=0.001,
              critic_lr_input=0.001,
              ):

        batch = self.replay_buffer.sample_batch(batch_size)
        feed_dict = {self.x_ph: batch['obs1'],
                     self.x2_ph: batch['obs2'],
                     self.a_ph: batch['acts'],
                     self.r_ph: batch['rews'],
                     self.d_ph: batch['done'],
                     self.actor_lr: actor_lr_input,
                     self.critic_lr: critic_lr_input,
                     }
        outs = self.sess.run(self.step_ops,
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
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=random_seed)
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default='sac_auto_class')
    args = parser.parse_args()

    env = gym.make(args.env)
    env = env.unwrapped
    env.seed(args.seed)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high[0]

    net = SAC(a_dim, s_dim, a_bound,
            #   batch_size=100,
              sess_opt=0.1
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
                a = net.get_action(s, 0.1)

            a = np.clip(a, -a_bound, a_bound)

            s_, r, done, info = env.step(a)
            done = False if j == args.max_steps - 1 else done

            net.store_transition((s, a, r, s_, done))

            s = s_
            ep_reward += r
            if j == args.max_steps - 1:
                ep_update_time = time.time()
                for _ in range(args.max_steps):
                    net.learn()
                ep_update_time = time.time() - ep_update_time
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