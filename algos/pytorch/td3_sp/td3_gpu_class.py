from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import algos.pytorch.td3_sp.core as core
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).cuda() for k,v in batch.items()}


class TD3:
    def __init__(self, a_dim, obs_dim, a_bound,
                actor_critic=core.MLPActorCritic, 
                ac_kwargs=dict(), seed=0, 
                steps_per_epoch=4000, epochs=100, 
                replay_size=int(1e6), gamma=0.99, 
                polyak=0.995, pi_lr=1e-3, q_lr=1e-3, 
                act_noise=0.1, target_noise=0.2, 
                noise_clip=0.5, policy_delay=2, 
                logger_kwargs=dict(), save_freq=1):
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.learn_step = 0
        self.obs_dim = obs_dim
        self.act_dim = a_dim
        self.act_limit = a_bound
        self.policy_delay = policy_delay
        self.action_noise = act_noise
        
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.polyak = polyak
        self.policy_delay = policy_delay
        self.gamma = gamma

        # Share information about action space with policy architecture
        ac_kwargs['action_space'] = a_bound

        # Create actor-critic module and target networks        
        self.ac = actor_critic(obs_dim=self.obs_dim, act_dim=self.act_dim, act_bound=self.act_limit).cuda()
        self.ac_targ = deepcopy(self.ac).cuda()

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False            
        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=replay_size)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q_optimizer = Adam(self.q_params, lr=q_lr)

    def get_action(self, s, noise_scale=0):
        s_cuda = torch.as_tensor(s, dtype=torch.float32).cuda()
        a = self.ac.act(s_cuda)
        a = a.detach().cpu().numpy()
        a += noise_scale * np.random.randn(self.act_dim)
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
                a = self.get_action(s)                
                s, r, d, _ = env.step(a)
                ep_reward += r
            ep_reward_list.append(ep_reward)
        mean_ep_reward = np.mean(np.array(ep_reward_list))
        return mean_ep_reward

    def compute_loss_pi(self, data):
        o = data['obs']
        q1_pi = self.ac.q1(o, self.ac.pi(o))
        return -q1_pi.mean()

    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)
        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ = self.ac_targ.pi(o2)
            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * self.target_noise
            epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -self.act_limit, self.act_limit)
            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * q_pi_targ
        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2
        # Useful info for logging
        # loss_info = dict(Q1Vals=q1.detach().numpy(),
        #                  Q2Vals=q2.detach().numpy())
        # loss_info = dict(Q1Vals=q1.detach().cpu().numpy(),
        #                  Q2Vals=q2.detach().cpu().numpy())
        loss_info = dict(Q1Vals=q1,
                         Q2Vals=q2)
        return loss_q, loss_info        

    def learn(self, batch_size=100,
              actor_lr_input=0.001,
              critic_lr_input=0.001,
              ):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        data = self.replay_buffer.sample_batch(batch_size)
        loss_q, loss_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()
        # Possibly update pi and target networks
        if self.learn_step % self.policy_delay == 0:            
            for p in self.q_params:
                p.requires_grad = False
            # Next run one gradient descent step for pi.
            self.pi_optimizer.zero_grad()
            loss_pi = self.compute_loss_pi(data)
            loss_pi.backward()
            self.pi_optimizer.step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in self.q_params:
                p.requires_grad = True

            with torch.no_grad():
                for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)
        self.learn_step += 1


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
    parser.add_argument('--exp_name', type=str, default='td3_torch_class')
    args = parser.parse_args()

    env = gym.make(args.env)
    env = env.unwrapped
    env.seed(args.seed)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high[0]

    net = TD3(a_dim, s_dim, a_bound,                            
              )
    ep_reward_list = []
    test_ep_reward_list = []

    for i in range(args.epochs):
        s = env.reset()
        ep_reward = 0
        st = time.time()
        for j in range(args.max_steps):
            # Add exploration noise
            # if i < 10:
            #     a = np.random.rand(a_dim) * a_bound
            # else:
            #     # a = net.choose_action(s)
            a = net.get_action(s, 0.1)            
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
