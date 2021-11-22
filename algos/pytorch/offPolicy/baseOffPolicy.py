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
        if not noise_scale:
            noise_scale = self.action_noise
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
    def save_episode(self, episode_trans, reward_func,):
        ep_obs = np.array([np.concatenate((trans[0]['observation'],
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
            obs_arr, next_obs_arr = map(self.convert_dict_to_array,
                                        (obs, next_obs))
            try:
                obs_arr = self.norm.normalize(v=obs_arr)
                next_obs_arr = self.norm.normalize(v=next_obs_arr)
            except:
                pass

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
                obs_arr, next_obs_arr = map(self.convert_dict_to_array,
                                            (obs, next_obs))
                try:
                    obs_arr = self.norm.normalize(v=obs_arr)
                    next_obs_arr = self.norm.normalize(v=next_obs_arr)
                except:
                    pass
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

    def convert_dict_to_array(self, obs_dict,
                              exclude_key=['achieved_goal']):
        obs_array = np.concatenate([obs_dict[key] for key, value in obs_dict.items() if key not in exclude_key])
        return obs_array

    # end HER utils

    def test_agent(self, args, env, n=5, logger=None, obs2state=None):
        ep_reward_list = []
        for j in range(n):
            obs = env.reset()
            ep_reward = 0
            success = []
            for i in range(args.n_steps):
                s = obs2state(obs)
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
        path = 'dense_replay.pkl'

        :param path: Path to the file where the replay buffer should be saved.
            if path is a str or pathlib.Path, the path is automatically created if necessary.
        """
        with open(path, 'wb') as f:
            pickle.dump(obj=self.replay_buffer, file=f,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def load_replay_buffer(self, path):
        """
        Load a replay buffer from a pickle file.

        :param path: Path to the pickled replay buffer.
        """        
        self.replay_buffer = pickle.load(open(path, 'rb'))


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
