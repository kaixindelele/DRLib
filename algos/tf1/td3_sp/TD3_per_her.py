import numpy as np
import tensorflow as tf
from algos.tf1.offPolicy.baseOffPolicy import OffPolicy
from algos.tf1.td3_sp import core
from algos.tf1.td3_sp.core import get_vars


class TD3(OffPolicy):
    def __init__(self,
                 act_dim, obs_dim, a_bound,
                 mlp_actor_critic=core.mlp_actor_critic,
                 ac_kwargs=dict(),
                 seed=0,
                 replay_size=int(1e6),
                 gamma=0.9,
                 polyak=0.99,
                 pi_lr=1e-3, q_lr=1e-3,
                 act_noise=0.1, target_noise=0.2,
                 noise_clip=0.5, policy_delay=2,
                 sess_opt=None,
                 sess=None,
                 batch_size=256,
                 buffer=None,
                 per_flag=True,
                 her_flag=True,
                 goal_selection_strategy="future",
                 n_sampled_goal=4,
                 action_l2=0.0,
                 clip_return=None,
                 state_norm=True,
                 ):
        super(TD3, self).__init__(act_dim, obs_dim, a_bound,
        mlp_actor_critic=core.mlp_actor_critic,
        ac_kwargs=ac_kwargs, seed=seed,
        replay_size=replay_size, gamma=gamma, polyak=polyak,
        pi_lr=pi_lr, q_lr=q_lr, batch_size=batch_size, act_noise=act_noise,
        target_noise=target_noise, noise_clip=noise_clip,
        policy_delay=policy_delay, sess_opt=sess_opt,
        per_flag=per_flag, her_flag=her_flag,
        goal_selection_strategy=goal_selection_strategy,
        n_sampled_goal=n_sampled_goal, action_l2=action_l2,
        clip_return=clip_return, state_norm=state_norm)

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
            a2 = tf.clip_by_value(a2, -self.a_bound, self.a_bound)

            # Target Q-values, using action from target policy
            _, q1_targ, q2_targ, _ = mlp_actor_critic(self.x2_ph, a2, **ac_kwargs)

        # Count variables
        var_counts = tuple(core.count_vars(scope)
                           for scope in ['main/pi',
                                         'main/q1',
                                         'main/q2',
                                         'main'])
        print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d, \t total: %d\n' % var_counts)

        # Bellman backup for Q functions, using Clipped Double-Q targets
        min_q_targ = tf.minimum(q1_targ, q2_targ)
        backup = tf.stop_gradient(self.r_ph + gamma * (1 - self.d_ph) * min_q_targ)
        # backup = self.r_ph + gamma * min_q_targ

        # TD3 losses
        self.pi_loss = -tf.reduce_mean(self.q1_pi)
        if action_l2:
            self.pi_loss += self.action_l2 * \
                        tf.reduce_mean(tf.square(self.pi / self.a_bound))
        q1_loss = tf.reduce_mean((self.q1 - backup) ** 2)
        q2_loss = tf.reduce_mean((self.q2 - backup) ** 2)

        if self.per_flag:
            self.abs_errors = tf.abs(backup - self.q1)
            self.q_loss = self.ISWeights * (q1_loss + q2_loss)
        else:
            # 正常的！
            self.q_loss = q1_loss + q2_loss

        # Separate train ops for pi, q
        pi_optimizer = tf.train.AdamOptimizer(learning_rate=self.actor_lr)
        q_optimizer = tf.train.AdamOptimizer(learning_rate=self.critic_lr)
        self.train_pi_op = pi_optimizer.minimize(self.pi_loss,
                                                 var_list=get_vars('main/pi'))
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

    # 删掉get_action，直接继承父类的

    def get_q(self, s, a):
        Q = self.sess.run([self.q1, self.q2],
                          feed_dict={self.x_ph: s.reshape(1, -1),
                                     self.a_ph: a.reshape(1, -1),
                                     })
        q1, q2 = Q[0][0], Q[1][0]
        return q1, q2

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


if __name__ == '__main__':
    pass
