import numpy as np
import gym
import os, sys
import torch
from mpi4py import MPI
from subprocess import CalledProcessError

import time
from spinup_utils.logx import setup_logger_kwargs, colorize
from spinup_utils.logx import EpochLogger
from spinup_utils.print_logger import Logger
from spinup_utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)
but I ignore it~
"""


def add_noise_obs2state(obs, args):
    # 给奖励函数加噪声
    noise_range = args.params_dict['nr']
    obj_pos = obs['achieved_goal']
    grip_pos = obs['grip_goal']
    noise_obj_pos = obj_pos + np.random.uniform(-noise_range, noise_range, size=3)
    noise_object_rel_pos = noise_obj_pos - grip_pos
    noise_observation = np.concatenate([obs['observation'][:3],
                                        noise_obj_pos,
                                        noise_object_rel_pos,
                                        obs['observation'][9:]
                                        ])
    obs.update({"noise_observation": noise_observation})
    obs.update({"noise_achieved_goal": noise_obj_pos})
    return obs


def obs2state(obs, key_list=['observation', 'desired_goal']):
    s = np.concatenate(([obs[key] for key in key_list]
                       ))
    return s


def trainer(dense_net, sparse_net, env, args, obj_num=3):
    # 标注奖励函数类型
    reward_type_dict = {0: "sparse",
                        1: "dense",
                        2: "double_dense",
                        3: "pos_tanh_dense",
                        4: "pos_double_dense",
                        5: "pos_sparse",
                        }
    # logger
    net_name = str(type(dense_net)).split('.')[-1][:-2]
    print("net_name：", net_name)
    env_name = args.params_dict['env']
    exp_name = args.exp_name + '_' + net_name + '-env-' + env_name
    rf_size = args.params_dict['rf']
    rf_size = "%.3g" % rf_size if hasattr(rf_size, "__float__") else rf_size
    exp_name += '-rf-' + rf_size
    if args.params_dict['d2s'] == 500:
        args.n_epochs = 600

    exclude_list = ['sd', 'rf', 'env', 're', 'rp', 'rn', 'bs', 'gamma']
    for key, value in args.params_dict.items():
        print("key, value:", key, value)
        if key not in exclude_list:
            exp_name += '_' + key + str(value).replace('.', '_')

    logger_kwargs = setup_logger_kwargs(exp_name=exp_name,
                                        seed=args.params_dict['sd'],
                                        output_dir=args.output_dir + "/",
                                        tune=True)
    logger = EpochLogger(**logger_kwargs)
    sys.stdout = Logger(logger_kwargs["output_dir"] + "/print.log",
                        sys.stdout)
    logger.save_config(locals(), tune=True, root_dir=__file__)
    start_time = time.time()
    dense_reward_list = [0]
    sparse_reward_list = [0]
    test_success_rates = []
    test_success_rate = 0.0

    for i in range(args.n_epochs):
        for c in range(args.n_cycles):
            obs = env.reset()
            obs = add_noise_obs2state(obs, args)
            episode_trans = []
            if args.params_dict['gd'] == 1:
                s = obs2state(obs, key_list=['noise_observation', 'desired_goal'])
            else:
                s = obs['noise_observation']
            ep_reward = 0
            real_ep_reward = 0
            episode_time = time.time()

            success = []
            for j in range(args.n_steps):
                if i >= args.params_dict['d2s']:
                    a = sparse_net.get_action(s, noise_scale=args.noise_ps)
                else:
                    a = dense_net.get_action(s, noise_scale=args.noise_ps)

                if np.random.random() < args.random_eps:
                    a = np.random.uniform(low=-dense_net.a_bound,
                                          high=dense_net.a_bound,
                                          size=dense_net.act_dim)

                a = np.clip(a, -dense_net.a_bound, dense_net.a_bound)
                # ensure the gripper close!
                try:
                    obs_next, r, done, info = env.step(a)
                    # print("reward:", r, 'goal:', obs_next['desired_goal'])
                    success.append(info["is_success"])
                except Exception as e:

                    print("Exception:", e)
                    success.append(int(done))
                    break

                obs_next = add_noise_obs2state(obs_next, args)
                if args.params_dict['gd'] == 1:
                    s_ = obs2state(obs_next, key_list=['noise_observation', 'desired_goal'])
                else:
                    s_ = obs_next['noise_observation']

                # visualization
                if args.params_dict['re']:
                    env.render()

                # 防止gym中的最大step会返回done=True
                done = False if j == args.n_steps - 1 else done

                if not args.params_dict['gd']:
                    dd_noise_r = env.compute_reward(obs_next['noise_achieved_goal'],
                                                    obs['desired_goal'], obs_next['grip_goal'],
                                                    reward_type=reward_type_dict[args.params_dict['drt']])
                    dense_net.store_transition((s, a, dd_noise_r, s_, done))

                    noise_r = env.compute_reward(obs_next['noise_achieved_goal'],
                                                 obs['desired_goal'], obs_next['grip_goal'],
                                                 reward_type=reward_type_dict[args.params_dict['srt']])
                    sparse_net.store_transition((s, a, noise_r, s_, done))

                episode_trans.append([obs, a, r, obs_next, done, info])
                s = s_
                obs = obs_next
                ep_reward += r
                real_ep_reward += r
            if args.params_dict['gd']:
                dense_net.save_episode(episode_trans=episode_trans,
                                       reward_func=env.compute_reward,
                                       obs2state=obs2state,
                                       args=args,
                                       reward_type=reward_type_dict[args.params_dict['drt']])
                sparse_net.save_episode(episode_trans=episode_trans,
                                        reward_func=env.compute_reward,
                                        obs2state=obs2state,
                                        args=args,
                                        reward_type=reward_type_dict[args.params_dict['srt']])

            logger.store(EpRet=ep_reward)
            logger.store(EpRealRet=real_ep_reward)

            if i > 0:
                for _ in range(args.params_dict['un']):
                    if i >= args.params_dict['d2s']:
                        # sparse_net.positive_sparse = True
                        outs = sparse_net.learn(args.params_dict['bs'],
                                                args.base_lr,
                                                args.base_lr,
                                                )
                    else:
                        outs = dense_net.learn(args.params_dict['bs'],
                                               args.base_lr,
                                               args.base_lr,
                                               )

                    if outs[1] is not None:
                        logger.store(Q1=outs[1])
                        logger.store(Q2=outs[2])
            else:
                logger.store(Q1=0.0)
                logger.store(Q2=0.0)

            if 0.0 < sum(success) < args.n_steps:
                print("epoch:", i,
                      "\tep:", c,
                      "\tep_rew:", ep_reward,
                      "\ttime:", np.round(time.time()-episode_time, 3),
                      '\tdone:', sum(success),
                      "\tenv_name:", env_name,)
        if i >= args.params_dict['d2s']:
            test_ep_reward, logger = sparse_net.test_agent(args=args,
                                                           env=env,
                                                           n=10,
                                                           logger=logger,
                                                           obs2state=obs2state,
                                                           add_noise_obs2state=add_noise_obs2state,
                                                           )
            dense_reward_list.append(dense_reward_list[-1])
            sparse_reward_list.append(np.mean(logger.epoch_dict['TestSuccess']))
        else:
            test_ep_reward, logger = dense_net.test_agent(args=args,
                                                          env=env,
                                                          n=10,
                                                          logger=logger,
                                                          obs2state=obs2state,
                                                          add_noise_obs2state=add_noise_obs2state,
                                                          )
            dense_reward_list.append(np.mean(logger.epoch_dict['TestSuccess']))
            sparse_reward_list.append(0.0)

        logger.store(TestEpRet=test_ep_reward)
        # reset this sparse model.
        test_success_rates.append(np.mean(np.array(logger.epoch_dict['TestSuccess'])))
        print("test_success_rates:", test_success_rates)
        if test_success_rates[-1] < 0.8 and i % args.params_dict['d2s'] == 0:
            sparse_net.reset_net()
            print("test_success_rates:", test_success_rates)
            print("epoch:", i)
            print("sparse_net.reset_net()!!!!!!!!!!!!!!!!!")

        logger.log_tabular('Epoch', i)
        logger.log_tabular('EpRet', average_only=True)
        logger.log_tabular('EpRealRet', average_only=True)
        logger.log_tabular('TestEpRet', average_only=True)

        logger.log_tabular('Q1', with_min_and_max=True)
        logger.log_tabular('Q2', average_only=True)

        logger.log_tabular('TestSuccess', average_only=True)

        logger.log_tabular('TotalEnvInteracts', i * args.n_cycles * args.n_steps + c * args.n_steps + j + 1)
        logger.log_tabular('TotalTime', time.time() - start_time)
        logger.log_tabular('Dense2Sparse', args.params_dict['d2s'])
        logger.dump_tabular()

    print(colorize("the experience %s is end" % logger.output_file.name,
                   'green', bold=True))


def launch(net, thunk_params_dict_list, args):
    p_id = proc_id()
    print("p_id:", p_id)
    if p_id > len(thunk_params_dict_list) - 1:
        print("p_id:", p_id)
        print("sys.exit()")
        sys.exit()
    params_dict = thunk_params_dict_list[p_id]
    args.params_dict = params_dict

    # 确保不同进程的随机种子不同！
    if 'sd' in args.params_dict.keys():
        args.seed = args.params_dict["sd"]

    if 'FetchDoublePush-v1' in args.params_dict['env']:
        from gym.envs.robotics.fetch.dpush import FetchDoublePushEnv
        env = FetchDoublePushEnv()
        obj_num = 2

    elif 'FetchStack-v1' in args.params_dict['env']:
        from gym.envs.robotics.fetch.stack import FetchStackEnv
        env = FetchStackEnv()
        obj_num = 2

    elif 'FetchDrawerBox-v1' in args.params_dict['env']:
        from gym.envs.robotics.fetch.drawer_box import FetchDrawerBoxEnv
        env = FetchDrawerBoxEnv()
        obj_num = 2

    elif 'FetchThreeStack-v1' in args.params_dict['env']:
        from gym.envs.robotics.fetch.tstack import FetchThreeStackEnv
        env = FetchThreeStackEnv()
        obj_num = 3
    elif 'FetchThreePush-v1' in args.params_dict['env']:
        from gym.envs.robotics.fetch.tpush import FetchThreePushEnv
        env = FetchThreePushEnv()
        obj_num = 3
    elif 'FetchPickAndPlace-v1' in args.params_dict['env']:
        from gym.envs.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv
        env = FetchPickAndPlaceEnv()
        obj_num = 1
    elif 'FetchPush-v1' in args.params_dict['env']:
        from gym.envs.robotics.fetch.push import FetchPushEnv
        env = FetchPushEnv()
        obj_num = 1
    else:
        env = gym.make(args.params_dict['env'])
        obj_num = 1

    env.reward_type = 'pos_double_dense'

    # set goal distribution and observation type.
    if args.params_dict['gd'] == 0:
        env.goal_distribution = 0
        s_dim = env.observation_space.spaces['observation'].shape[0]
    else:
        env.goal_distribution = 1
        s_dim = env.observation_space.spaces['observation'].shape[0] + \
                env.observation_space.spaces['desired_goal'].shape[0]

    env.seed(args.seed)
    np.random.seed(args.seed)

    act_dim = env.action_space.shape[0]
    a_bound = env.action_space.high[0]

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    device = torch.device("cuda:" + str(0))
    print("device:", device)

    dense_net = net(act_dim=act_dim,
                    obs_dim=s_dim,
                    a_bound=a_bound,
                    per_flag=args.per,
                    her_flag=args.her,
                    replay_size=int(args.params_dict['rf']),
                    action_l2=args.action_l2,
                    state_norm=args.state_norm,
                    gamma=float(args.params_dict['gamma']),
                    sess_opt=args.sess_opt,
                    seed=args.seed,
                    clip_return=args.clip_return,
                    device=device,
                    )
    sparse_net = net(act_dim=act_dim,
                     obs_dim=s_dim,
                     a_bound=a_bound,
                     per_flag=args.per,
                     her_flag=args.her,
                     replay_size=int(args.params_dict['rf']),
                     action_l2=args.action_l2,
                     state_norm=args.state_norm,
                     gamma=float(args.params_dict['gamma']),
                     sess_opt=args.sess_opt,
                     seed=args.seed,
                     clip_return=args.clip_return,
                     device=device,
                     )
    trainer(dense_net, sparse_net, env, args, obj_num=obj_num)


if __name__ == '__main__':
    pass