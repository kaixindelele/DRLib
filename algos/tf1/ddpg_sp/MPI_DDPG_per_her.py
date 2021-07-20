import numpy as np
import tensorflow as tf
import gym
import time
import sys
import copy
from algos.tf1.offPolicy.baseOffPolicy import OffPolicy
from algos.tf1.ddpg_sp import core
from algos.tf1.ddpg_sp.core import get_vars

# 在tf中，MPI只需要将优化器替换掉，替换成能求各个进程平均梯度的优化器。
from spinup_utils.mpi_tf import MpiAdamOptimizer
import warnings
warnings.filterwarnings("ignore")


class DDPG(OffPolicy):
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
        super(DDPG, self).__init__(act_dim, obs_dim, a_bound,
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
            self.pi, self.q, q_pi = mlp_actor_critic(self.x_ph, self.a_ph, **ac_kwargs)

        # Target networks
        with tf.variable_scope('target'):
            # Note that the action placeholder going to actor_critic here is
            # irrelevant, because we only need q_targ(s, pi_targ(s)).
            pi_targ, _, q_pi_targ = mlp_actor_critic(self.x2_ph, self.a_ph, **ac_kwargs)

        # Count variables
        var_counts = tuple(core.count_vars(scope) for scope in ['main/pi', 'main/q', 'main'])
        print('\nNumber of parameters: \t pi: %d, \t q: %d, \t total: %d\n' % var_counts)

        # Bellman backup for Q function
        backup = tf.stop_gradient(self.r_ph + gamma * (1 - self.d_ph) * q_pi_targ)
        # clip return
        # 当用于HER的时候，奖励值为{-1, 0}时，可以用下面的设置，如果奖励函数变了，记得修改
        clip_return = 1 / (1 - gamma)
        backup = tf.clip_by_value(backup, -clip_return, 0)

        # DDPG losses
        self.pi_loss = -tf.reduce_mean(q_pi)
        self.pi_loss += self.action_l2 * \
                        tf.reduce_mean(tf.square(self.pi / self.a_bound))

        if self.per_flag:
            # q_target - q
            self.abs_errors = tf.abs(backup - self.q)
            self.q_loss = self.ISWeights * tf.reduce_mean((self.q - backup) ** 2)
        else:
            # 正常的！
            self.q_loss = tf.reduce_mean((self.q - backup) ** 2)

        # Separate train ops for pi, q
        pi_optimizer = MpiAdamOptimizer(learning_rate=self.actor_lr)
        q_optimizer = MpiAdamOptimizer(learning_rate=self.critic_lr)
        self.train_pi_op = pi_optimizer.minimize(self.pi_loss, var_list=get_vars('main/pi'))
        self.train_q_op = q_optimizer.minimize(self.q_loss, var_list=get_vars('main/q'))

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

    def learn(self, batch_size=100, actor_lr_input=0.001,
              critic_lr_input=0.001,):
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
            q_step_ops = [self.q_loss, self.q,
                          self.train_q_op,
                          self.abs_errors,
                          ]
            outs = self.sess.run(q_step_ops, feed_dict)
            q_loss, q, train_q_op, abs_errors = outs
            if self.learn_step % self.policy_delay == 0:
                # Delayed policy update
                pi_outs = self.sess.run([self.pi_loss,
                                        self.train_pi_op,
                                        self.target_update],
                                        feed_dict)

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
            q_step_ops = [self.train_q_op]

            # Q-learning update
            outs = self.sess.run([self.q_loss, self.q, self.train_q_op],
                                 feed_dict)
            # Policy update
            pi_outs = self.sess.run([self.pi_loss, self.train_pi_op,
                                     self.target_update,
                                     ],
                                    feed_dict)

            self.learn_step += 1
            return outs

    def update_target(self):
        # Policy update
        self.sess.run([self.target_update])

