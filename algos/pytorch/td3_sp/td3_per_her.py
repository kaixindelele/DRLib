from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import algos.pytorch.td3_sp.core as core
from algos.pytorch.offPolicy.baseOffPolicy import OffPolicy


class TD3Torch(OffPolicy):
    def __init__(self,
                 act_dim, obs_dim, a_bound,
                 actor_critic=core.MLPActorCritic,
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
                 device=None,
                 ):
        super(TD3Torch, self).__init__(act_dim, obs_dim, a_bound,
                                  actor_critic=core.MLPActorCritic,
                                  ac_kwargs=ac_kwargs, seed=seed,
                                  replay_size=replay_size, gamma=gamma, polyak=polyak,
                                  pi_lr=pi_lr, q_lr=q_lr, batch_size=batch_size, act_noise=act_noise,
                                  target_noise=target_noise, noise_clip=noise_clip,
                                  policy_delay=policy_delay, sess_opt=sess_opt,
                                  per_flag=per_flag, her_flag=her_flag,
                                  goal_selection_strategy=goal_selection_strategy,
                                  n_sampled_goal=n_sampled_goal, action_l2=action_l2,
                                  clip_return=clip_return, state_norm=state_norm,
                                  device=device)

        # Create actor-critic module and target networks
        self.ac = actor_critic(obs_dim=self.obs_dim, act_dim=self.act_dim, act_bound=self.a_bound).to(self.device)
        self.ac_targ = deepcopy(self.ac).to(self.device)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False            
        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q_optimizer = Adam(self.q_params, lr=q_lr)

    def compute_loss_pi(self, data):
        if self.per_flag:
            tree_idx, batch_memory, ISWeights = data
            o = []
            for i in range(len(batch_memory)):
                o.append(batch_memory[i][0])
            o = torch.as_tensor(np.array(o), dtype=torch.float32, device=self.device)
        else:
            o = data['obs']

        q1_pi = self.ac.q1(o, self.ac.pi(o))
        return -q1_pi.mean()

    def compute_loss_q(self, data):
        if self.per_flag:
            tree_idx, batch_memory, ISWeights = data
            o, a, r, o2, d = [], [], [], [], []
            for i in range(len(batch_memory)):
                o.append(batch_memory[i][0])
                a.append(batch_memory[i][1])
                r.append(batch_memory[i][2])
                o2.append(batch_memory[i][3])
                d.append(batch_memory[i][4])
            o = torch.as_tensor(np.array(o), dtype=torch.float32, device=self.device)
            a = torch.as_tensor(np.array(a), dtype=torch.float32, device=self.device)
            r = torch.as_tensor(np.array(r), dtype=torch.float32, device=self.device)
            o2 = torch.as_tensor(np.array(o2), dtype=torch.float32, device=self.device)
            d = torch.as_tensor(np.array(d), dtype=torch.float32, device=self.device)
            ISWeights = torch.as_tensor(np.array(ISWeights), dtype=torch.float32, device=self.device)
        else:
            o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)
        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ = self.ac_targ.pi(o2)
            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * self.target_noise
            epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -self.a_bound, self.a_bound)
            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2
        loss_info = dict(Q1Vals=q1,
                         Q2Vals=q2)
        if self.per_flag:
            loss_q = (ISWeights * ((q1 - backup)**2 + (q2 - backup)**2)).mean()
            abs_errors = torch.abs(backup - (q1+q2)/2)
            loss_info['abs_errors'] = abs_errors.detach().cpu().numpy()
            loss_info['tree_idx'] = tree_idx
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
        if self.per_flag:
            self.replay_buffer.batch_update(tree_idx=loss_info['tree_idx'],
                                            abs_errors=loss_info['abs_errors'])  # update priority
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
        # print("loss_info:", loss_info)
        return loss_q, loss_info['Q1Vals'].detach().cpu().numpy(), loss_info['Q2Vals'].detach().cpu().numpy()

