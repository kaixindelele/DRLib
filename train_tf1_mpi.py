import numpy as np
import gym
import os, sys
from arguments import get_args
from mpi4py import MPI
from subprocess import CalledProcessError

import time
from spinup_utils.logx import setup_logger_kwargs, colorize
from spinup_utils.logx import EpochLogger
from spinup_utils.print_logger import Logger
from spinup_utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

from spinup_utils.mpi_tf import MpiAdamOptimizer, sync_all_params
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)
but I ignore it~

"""


def obs2state(obs, key_list=['observation', 'desired_goal']):
    if type(obs) == dict:
        s = np.concatenate(([obs[key] for key in key_list]
                            ))
    elif type(obs) == np.ndarray:
        s = obs[:]
    else:
        s = obs[:]
    return s


def trainer(net, env, args):
    # logger
    exp_name = args.exp_name+'_'+args.RL_name+'_'+args.env_name
    logger_kwargs = setup_logger_kwargs(exp_name=exp_name,
                                        seed=args.seed,
                                        output_dir=args.output_dir + "/")
    logger = EpochLogger(**logger_kwargs)
    
    if proc_id() == 0:
        sys.stdout = Logger(logger_kwargs["output_dir"] + "/print.log",
                            sys.stdout)
        logger.save_config(locals(), __file__)
    # start trainning
    start_time = time.time()
    for i in range(args.n_epochs):
        for c in range(args.n_cycles):
            obs = env.reset()
            episode_trans = []
            s = obs2state(obs)
            ep_reward = 0
            real_ep_reward = 0
            episode_time = time.time()

            success = []
            for j in range(args.n_steps):
                a = net.get_action(s, noise_scale=args.noise_ps)
                # a = net.get_action(s)
                # a = noise.add_noise(a)

                if np.random.random() < args.random_eps:
                    a = np.random.uniform(low=-net.a_bound,
                                          high=net.a_bound,
                                          size=net.act_dim)
                a = np.clip(a, -net.a_bound, net.a_bound)

                try:
                    obs_next, r, done, info = env.step(a)
                    success.append(info["is_success"])
                except Exception as e:
                    success.append(int(done))
                s_ = obs2state(obs_next)

                # visualization
                if args.render and i % 3 == 0 and c % 20 == 0:
                    env.render()
                done = False if j == args.n_steps - 1 else done
                if not args.her:
                    net.store_transition((s, a, r, s_, done))

                episode_trans.append([obs, a, r, obs_next, done, info])
                s = s_
                obs = obs_next
                ep_reward += r
                real_ep_reward += r
            if args.her:
                net.save_episode(episode_trans=episode_trans,
                                 reward_func=env.compute_reward)
            logger.store(EpRet=ep_reward)
            logger.store(EpRealRet=real_ep_reward)

            for _ in range(40):
                outs = net.learn(args.batch_size,
                                 args.base_lr,
                                 args.base_lr * 2,
                                 )
                if outs[1] is not None:
                    logger.store(Q1=outs[1])
                    logger.store(Q2=outs[2])
            if 0.0 < sum(success) < args.n_steps:
                print("epoch:", i,
                      "\tep:", c,
                      "\tep_rew:", ep_reward,
                      "\ttime:", np.round(time.time()-episode_time, 3),
                      '\tdone:', sum(success))

        test_ep_reward, logger = net.test_agent(args=args,
                                                env=env,
                                                n=10,
                                                logger=logger,
                                                obs2state=obs2state,
                                                )
        logger.store(TestEpRet=test_ep_reward)

        logger.log_tabular('Epoch', i)
        logger.log_tabular('EpRet', average_only=True)
        logger.log_tabular('EpRealRet', average_only=True)
        logger.log_tabular('TestEpRet', average_only=True)

        logger.log_tabular('Q1', with_min_and_max=True)
        logger.log_tabular('Q2', average_only=True)

        logger.log_tabular('TestSuccess', average_only=True)

        logger.log_tabular('TotalEnvInteracts', i * args.n_cycles * args.n_steps + c * args.n_steps + j + 1)
        logger.log_tabular('TotalTime', time.time() - start_time)
        logger.dump_tabular()

    print(colorize("the experience %s is end" % logger.output_file.name,
                   'green', bold=True))
    net.save_simple_network(logger_kwargs["output_dir"])


def launch(net, args):
    env = gym.make(args.env_name)
    # 确保不同进程的随机种子不同！
    seed = args.seed
    seed += 10000 * proc_id()    
    env.seed(seed)
    np.random.seed(seed)    

    try:
        s_dim = env.observation_space.shape[0]
    except:
        s_dim = env.observation_space.spaces['observation'].shape[0] + \
                env.observation_space.spaces['desired_goal'].shape[0]

    act_dim = env.action_space.shape[0]
    a_bound = env.action_space.high[0]

    net = net(act_dim=act_dim,
              obs_dim=s_dim,
              a_bound=a_bound,
              per_flag=args.per,
              her_flag=args.her,
              action_l2=args.action_l2,
              state_norm=args.state_norm,
              gamma=args.gamma,
              sess_opt=args.sess_opt,
              seed=args.seed,
              clip_return=args.clip_return,
              )
    # 同步强化网络的参数，如果网络不仅有强化的，可能会出问题；
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # Sync params across processes
    sess.run(sync_all_params())

    trainer(net, env, args)


if __name__ == '__main__':
    # os.environ['OMP_NUM_THREADS'] = '1'
    # os.environ['MKL_NUM_THREADS'] = '1'
    # 如果CPU比较多，GPU不够的话，建议只使用CPU~
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # get the params
    args = get_args()
    # run parallel code with mpi
    mpi_fork(args.cpu)
    from algos.tf1.td3_sp.MPI_TD3_per_her import TD3
    from algos.tf1.ddpg_sp.MPI_DDPG_per_her import DDPG
    from algos.tf1.sac_sp.MPI_SAC_per_her import SAC
    from algos.tf1.sac_auto.MPI_sac_auto_per_her import SAC_AUTO
    RL_list = [TD3, DDPG, SAC, SAC_AUTO]

    [launch(net=net, args=args) for net in RL_list if net.__name__ == args.RL_name]
