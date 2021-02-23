import numpy as np
import tensorflow as tf
import gym
import time
import sys
sys.path.append("../")
try:
    from rl_algorithms.sac_sp import core
    from rl_algorithms.sac_sp.core import get_vars
except:
    from sac_sp import core
    from sac_sp.core import get_vars


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
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
                 polyak=0.995, alpha=0.2,
                 pi_lr=1e-3, q_lr=1e-3,
                 batch_size=100,
                 # start_steps=10000,
                 act_noise=0.1, target_noise=0.2,
                 noise_clip=0.5, policy_delay=2,
                 # max_ep_len=1000,
                 # logger_kwargs=dict(), save_freq=1
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
        self.x_ph, self.a_ph, self.x2_ph, self.r_ph, self.d_ph = core.placeholders(obs_dim, a_dim, obs_dim, None, None)

        # Main outputs from computation graph
        with tf.variable_scope('main'):
            self.mu, self.pi, logp_pi, q1, q2, q1_pi, q2_pi, v = mlp_actor_critic(self.x_ph, self.a_ph, **ac_kwargs)

        # Target value network
        with tf.variable_scope('target'):
            _, _, _, _, _, _, _, v_targ = mlp_actor_critic(self.x2_ph, self.a_ph, **ac_kwargs)

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=self.act_dim, size=replay_size)

        # Count variables
        var_counts = tuple(core.count_vars(scope) for scope in
                           ['main/pi', 'main/q1', 'main/q2', 'main/v', 'main'])
        print(('\nNumber of parameters: \t pi: %d, \t' + \
               'q1: %d, \t q2: %d, \t v: %d, \t total: %d\n') % var_counts)

        # Min Double-Q:
        min_q_pi = tf.minimum(q1_pi, q2_pi)

        # Targets for Q and V regression
        q_backup = tf.stop_gradient(self.r_ph + gamma * (1 - self.d_ph) * v_targ)
        v_backup = tf.stop_gradient(min_q_pi - alpha * logp_pi)

        # Soft actor-critic losses
        pi_loss = tf.reduce_mean(alpha * logp_pi - q1_pi)
        q1_loss = 0.5 * tf.reduce_mean((q_backup - q1) ** 2)
        q2_loss = 0.5 * tf.reduce_mean((q_backup - q2) ** 2)
        v_loss = 0.5 * tf.reduce_mean((v_backup - v) ** 2)
        value_loss = q1_loss + q2_loss + v_loss

        # Policy train op
        # (has to be separate from value train op, because q1_pi appears in pi_loss)
        pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
        self.train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))

        # Value train op
        # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
        value_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)
        value_params = get_vars('main/q') + get_vars('main/v')
        with tf.control_dependencies([self.train_pi_op]):
            self.train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

        # Polyak averaging for target variables
        # (control flow because sess.run otherwise evaluates in nondeterministic order)
        with tf.control_dependencies([self.train_value_op]):
            target_update = tf.group([tf.assign(v_targ, polyak * v_targ + (1 - polyak) * v_main)
                                      for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        # All ops to call during one training step
        self.step_ops = [pi_loss, q1_loss, q2_loss, v_loss, q1, q2, v, logp_pi,
                         self.train_pi_op, self.train_value_op, target_update]

        # Initializing targets to match main variables
        target_init = tf.group([tf.assign(v_targ, v_main)
                                for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(target_init)

    def get_action(self, s, noise_scale=0):
        if not noise_scale:
            act_op = self.mu
        else:
            act_op = self.pi
        a = self.sess.run(act_op, feed_dict={self.x_ph: s.reshape(1, -1)})[0]
        
        return np.clip(a, -self.act_limit, self.act_limit)

    def store_transition(self, transition):
        (s, a, r, s_, done) = transition
        self.replay_buffer.store(s, a, r, s_, done)

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

    def learn(self, batch_size=100):

        batch = self.replay_buffer.sample_batch(batch_size)
        feed_dict = {self.x_ph: batch['obs1'],
                     self.x2_ph: batch['obs2'],
                     self.a_ph: batch['acts'],
                     self.r_ph: batch['rews'],
                     self.d_ph: batch['done']
                     }
        outs = self.sess.run(self.step_ops,feed_dict)
        self.learn_step += 1

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=random_seed)
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default='sac_class')
    args = parser.parse_args()

    env = gym.make(args.env)
    env = env.unwrapped
    env.seed(args.seed)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high[0]

    net = SAC(a_dim, s_dim, a_bound,
              batch_size=100,
              )
    ep_reward_list = []
    test_ep_reward_list = []

    for i in range(args.epochs):
        s = env.reset()
        ep_reward = 0
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

                for _ in range(args.max_steps):
                    net.learn()

                ep_reward_list.append(ep_reward)
                print('Episode:', i, ' Reward: %i' % int(ep_reward),
                      # 'Explore: %.2f' % var,
                      "learn step:", net.learn_step)
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
