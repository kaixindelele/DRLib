import numpy as np
import tensorflow as tf
import gym
import time
import sys
from algos.tf1.offPolicy.baseOffPolicy import OffPolicy
from algos.tf1.sac_sp import core
from algos.tf1.sac_sp.core import get_vars


class SAC(OffPolicy):
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
                 alpha=0.2,
                 ):
        super(SAC, self).__init__(act_dim, obs_dim, a_bound,
                                  mlp_actor_critic=core.mlp_actor_critic,
                                  ac_kwargs=ac_kwargs, seed=seed,
                                  replay_size=replay_size,
                                  gamma=gamma, polyak=polyak,
                                  pi_lr=pi_lr, q_lr=q_lr,
                                  batch_size=batch_size, act_noise=act_noise,
                                  target_noise=target_noise,
                                  noise_clip=noise_clip,
                                  policy_delay=policy_delay,
                                  sess_opt=sess_opt,
                                  per_flag=per_flag, her_flag=her_flag,
                                  goal_selection_strategy=goal_selection_strategy,
                                  n_sampled_goal=n_sampled_goal,
                                  action_l2=action_l2,
                                  clip_return=clip_return,
                                  state_norm=state_norm)

        # Main outputs from computation graph
        with tf.variable_scope('main'):
            self.mu, self.pi, logp_pi, q1, q2, q1_pi, q2_pi, v = mlp_actor_critic(self.x_ph, self.a_ph, **ac_kwargs)

        # Target value network
        with tf.variable_scope('target'):
            _, _, _, _, _, _, _, v_targ = mlp_actor_critic(self.x2_ph, self.a_ph, **ac_kwargs)

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
        # value_loss = q1_loss + q2_loss + v_loss

        if self.per_flag:
            # 这里有点问题，我也不清楚该选啥作为td_error
            self.abs_errors = tf.abs(q_backup - q1)
            value_loss = self.ISWeights * (q1_loss + q2_loss + v_loss)
        else:
            # 正常的！
            value_loss = q1_loss + q2_loss + v_loss

        # Policy train op
        # (has to be separate from value train op, because q1_pi appears in pi_loss)
        pi_optimizer = tf.train.AdamOptimizer(learning_rate=self.actor_lr)
        self.train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))

        # Value train op
        # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
        value_optimizer = tf.train.AdamOptimizer(learning_rate=self.critic_lr)
        value_params = get_vars('main/q') + get_vars('main/v')
        with tf.control_dependencies([self.train_pi_op]):
            self.train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

        # Polyak averaging for target variables
        # (control flow because sess.run otherwise evaluates in nondeterministic order)
        with tf.control_dependencies([self.train_value_op]):
            target_update = tf.group([tf.assign(v_targ, polyak * v_targ + (1 - polyak) * v_main)
                                      for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        # All ops to call during one training step
        if self.per_flag:
            self.step_ops = [pi_loss, q1_loss, q2_loss,
                             v_loss, q1, q2, v, logp_pi,
                             self.train_pi_op,
                             self.train_value_op,
                             target_update,
                             self.abs_errors,
                             ]
        else:
            self.step_ops = [pi_loss, q1_loss, q2_loss,
                             v_loss, q1, q2, v, logp_pi,
                             self.train_pi_op,
                             self.train_value_op,
                             target_update]

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

    def learn(self,
              batch_size=100,
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

            outs = self.sess.run(self.step_ops,
                                 feed_dict)
            # abs_errors = outs[-1]
            pi_loss, q1_loss, q2_loss, q1, q2, logp_pi, alpha, _, _, _, train_alpha_op, abs_errors = outs
            self.replay_buffer.batch_update(tree_idx,
                                            abs_errors)  # update priority
            self.learn_step += 1
            return outs
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
            outs = self.sess.run(self.step_ops,feed_dict)
            self.learn_step += 1
            return outs
