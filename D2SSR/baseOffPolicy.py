import numpy as np
import torch
import copy
import pickle
from algos.pytorch.offPolicy.norm import StateNorm


class OffPolicy:
    def __init__(self,
                 act_dim, obs_dim, a_bound,                 
                 actor_critic=None,
                 ac_kwargs=dict(), seed=0,
                 replay_size=int(1e6), 
                 gamma=0.99,
                 polyak=0.995, 
                 pi_lr=1e-3, 
                 q_lr=1e-3,
                 batch_size=256,
                 act_noise=0.1, 
                 target_noise=0.2,
                 noise_clip=0.5,                 
                 policy_delay=2,
                 sess_opt=None,
                 per_flag=True,
                 her_flag=True,
                 goal_selection_strategy="future",
                 n_sampled_goal=4,
                 action_l2=1.0,
                 clip_return=None,
                 state_norm=True,
                 device=torch.device("cuda" if torch.cuda.is_available() else 'cpu'),
                 ):
        # torch setting
        torch.manual_seed(seed)
        self.device = device

        self.learn_step = 0

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.a_bound = a_bound
        self.policy_delay = policy_delay
        self.action_noise = act_noise
        self.gamma = gamma
        self.replay_size = replay_size
        self.polyak = polyak
        self.target_noise = target_noise
        self.noise_clip = noise_clip

        # Share information about action space with policy architecture
        ac_kwargs['action_space'] = a_bound
        self.ac_kwargs = ac_kwargs

        # Experience buffer
        self.per_flag = per_flag
        self.her_flag = her_flag
        self.goal_selection_strategy = goal_selection_strategy
        self.n_sampled_goal = n_sampled_goal
        self.state_norm = state_norm
        if self.state_norm:
            self.norm = StateNorm(size=self.obs_dim)
        self.action_l2 = action_l2
        self.clip_return = clip_return

        if self.per_flag:
            from memory.sp_per_memory_torch import ReplayBuffer
        else:
            from memory.sp_memory_torch import ReplayBuffer
        # from memory.sp_memory_torch import ReplayBuffer
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim,
                                          act_dim=self.act_dim,
                                          size=self.replay_size,
                                          device=self.device)

    def get_action(self, s, noise_scale=0):
        if self.norm is not None:
            s = self.norm.normalize(v=s)
        s_cuda = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = self.ac.act(s_cuda)
        a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, -self.a_bound, self.a_bound)

    def store_transition(self, transition):
        if self.per_flag:
            self.replay_buffer.store(transition)
        else:
            (s, a, r, s_, done) = transition
            self.replay_buffer.store(s, a, r, s_, done)

    # HER utils
    def save_episode(self, episode_trans, reward_func, obs2state, args, reward_type='sparse'):
        ep_obs = np.array([np.concatenate((trans[0]['noise_observation'],
                                           trans[0]['desired_goal'],
                                           )) for trans in episode_trans])
        self.norm.update(v=ep_obs)
        for transition_idx, transition in enumerate(episode_trans):
            # We cannot sample a goal from the future in the last step of an episode
            if (transition_idx == len(episode_trans) - 1 and
                    self.goal_selection_strategy == "future"):
                break
            obs, action, reward, next_obs, done, info = copy.deepcopy(transition)
            # 注意，字典转元组的函数中，需要设定你环境中特定的key！如果搞不定的话，直接用下面的语句替代：
            # obs_arr = np.concatenate([obs[key1], obs[key2]])
            obs_arr, next_obs_arr = map(obs2state,
                                        *zip([obs, ['noise_observation', 'desired_goal']],
                                             [next_obs, ['noise_observation', 'desired_goal']]))
            try:
                obs_arr = self.norm.normalize(v=obs_arr)
                next_obs_arr = self.norm.normalize(v=next_obs_arr)
            except:
                pass
            # recompute the reward by sparse function:
            reward = reward_func(next_obs['noise_achieved_goal'],
                                 obs['desired_goal'], next_obs['grip_goal'], reward_type=reward_type)

            self.store_transition(transition=(obs_arr, action, reward, next_obs_arr, done))
            # HER 操作！
            ag_indexes = self.get_ag_indexes(episode_transitions=episode_trans,
                                             transition_idx=transition_idx,
                                             n_sampled_goal=self.n_sampled_goal)
            # For each sampled goals, store a new transition
            for ag_index in ag_indexes:
                # Copy transition to avoid modifying the original one
                # 默认obs是字典格式
                obs, action, reward, next_obs, done, info = copy.deepcopy(transition)
                label_key = 'desired_goal'
                relabel_key = 'noise_achieved_goal'
                obs[label_key] = self.get_ag(ag_index=ag_index,
                                             episode_transitions=episode_trans, key=relabel_key)
                next_obs[label_key] = self.get_ag(ag_index=ag_index,
                                                  episode_transitions=episode_trans, key=relabel_key)
                # Update the reward according to the new desired goal
                reward = reward_func(next_obs['noise_achieved_goal'],
                                     obs['desired_goal'], next_obs['grip_goal'],
                                     reward_type=reward_type)
                # Can we use achieved_goal == desired_goal?
                if ag_index == transition_idx + 1:
                    done = True
                else:
                    done = False
                # Transform back to ndarrays
                # map(func, (param1, param2))
                obs_arr, next_obs_arr = map(obs2state, (obs, next_obs))
                try:
                    obs_arr = self.norm.normalize(v=obs_arr)
                    next_obs_arr = self.norm.normalize(v=next_obs_arr)
                except:
                    pass
                # Add artificial transition to the replay buffer
                self.store_transition(transition=(obs_arr, action, reward, next_obs_arr, done))

    def get_ag_indexes(self, episode_transitions, transition_idx, n_sampled_goal=4):
        ag_indexes = [self._sample_achieved_goal_index(episode_transitions, transition_idx)
                      for _ in range(n_sampled_goal)]
        return ag_indexes

    def get_ag(self, ag_index, episode_transitions, key='achieved_goal'):
        ag = episode_transitions[ag_index][0][key]
        return ag

    def _sample_achieved_goal_index(self, episode_transitions, transition_idx):
        selected_idx = np.random.choice(np.arange(transition_idx + 1, len(episode_transitions)))
        return selected_idx
    # end HER utils

    def test_agent(self, args, env, n=5, logger=None, obs2state=None, add_noise_obs2state=None):
        ep_reward_list = []
        for j in range(n):
            obs = env.reset()
            ep_reward = 0
            success = []
            for i in range(args.n_steps):
                obs = add_noise_obs2state(obs, args)
                # noise state oracle reward!
                if args.params_dict['gd']:
                    s = obs2state(obs, key_list=['noise_observation', 'desired_goal'])
                else:
                    s = obs['noise_observation']
                a = self.get_action(s)
                try:
                    obs, r, done, info = env.step(a)
                    success.append(info['is_success'])
                except Exception as e:
                    success.append(int(done))

                ep_reward += r

            if logger:
                logger.store(TestEpRet=ep_reward)
                logger.store(TestSuccess=success[-1])

            ep_reward_list.append(ep_reward)
        mean_ep_reward = np.mean(np.array(ep_reward_list))
        if logger:
            return mean_ep_reward, logger
        else:
            return mean_ep_reward

    def learn(self, batch_size=100, actor_lr_input=0.001,
              critic_lr_input=0.001,):
        pass

    def save_step_network(self, time_step, save_path):
        act_save_path = save_path + '/actor_'+str(time_step)+'.pth'
        torch.save(self.ac.state_dict(), act_save_path)
        print("save model to:", save_path)

    def load_simple_network(self, path):
        self.ac.load_state_dict(torch.load(path))
        self.ac_targ.load_state_dict(torch.load(path))
        print("restore model successful")

    def save_simple_network(self, save_path):
        act_save_path = save_path + '/actor.pth'
        torch.save(self.ac.state_dict(), act_save_path)
        print("save model to:", save_path)

    def save_replay_buffer(self, path):
        """
        Save the replay buffer as a pickle file.
        path = 'replay.pkl'

        :param path: Path to the file where the replay buffer should be saved.
            if path is a str or pathlib.Path, the path is automatically created if necessary.
        """
        with open(path + '/replay.pkl', 'wb') as f:
            pickle.dump(obj=self.replay_buffer, file=f,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def load_replay_buffer(self, path):
        """
        Load a replay buffer from a pickle file.

        :param path: Path to the pickled replay buffer.
        """        
        self.replay_buffer = pickle.load(open(path, 'rb'))

    def save_norm(self, path):
        """
        Save the replay buffer as a pickle file.
        path = 'norm.pkl'

        :param path: Path to the file where the replay buffer should be saved.
            if path is a str or pathlib.Path, the path is automatically created if necessary.
        """
        with open(path + "/norm.pkl", 'wb') as f:
            pickle.dump(obj=self.norm, file=f,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def load_norm(self, path):
        """
        Load last norm from a pickle file.
        path = 'norm.pkl'

        :param path: Path to the pickled replay buffer.
        """
        self.norm = pickle.load(open(path, 'rb'))


if __name__ == '__main__':    
    net = TD3()
    logger_kwargs = {'output_dir':"logger/"}
    try:
        import os
        os.mkdir(logger_kwargs['output_dir'])
    except:
        pass
    # save buffer to local as .pkl
    path = logger_kwargs["output_dir"]+'/dense_'+str(args.seed)+'replay.pkl'
    net.save_replay_buffer(path)
    
    # load buffer from local .pkl 
    path = logger_kwargs["output_dir"]+'/dense_'+str(args.seed)+'replay.pkl'    
    net.load_replay_buffer(path)
