import argparse

"""
Here are the param for the training

"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='FetchPickAndPlace-v1', help='the environment name')
    parser.add_argument('--RL_name', type=str, default='TD3Torch', help='the RL name')
    parser.add_argument('--n_epochs', type=int, default=150, help='the number of epochs to train the agent')
    parser.add_argument('--n_cycles', type=int, default=50, help='the times to collect samples per epoch')
    parser.add_argument('--n_steps', type=int, default=50)

    parser.add_argument('--n_batches', type=int, default=40, help='the times to update the network')
    parser.add_argument('--save_interval', type=int, default=5, help='the interval that save the trajectory')
    parser.add_argument('--seed', type=int, default=300, help='random seed')
    gpu_id = 3
    cpu_num = 1
    parser.add_argument('--cpu', type=int, default=cpu_num, help='the number of cpus to collect samples')
    parser.add_argument('--replay_strategy', type=str, default='future', help='the HER strategy')
    parser.add_argument('--clip_return', type=float, default=50, help='if clip the returns')
    parser.add_argument('--save_dir', type=str, default='saved_models/', help='the path to save the models')
    parser.add_argument('--noise_ps', type=float, default=0.2, help='noise eps')
    parser.add_argument('--random_eps', type=float, default=0.3, help='random eps')
    parser.add_argument('--buffer_size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--replay_k', type=int, default=4, help='ratio to be replace')
    parser.add_argument('--clip_obs', type=float, default=200, help='the clip ratio')
    parser.add_argument('--batch_size', type=int, default=256, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor')
    parser.add_argument('--action_l2', type=float, default=1.0, help='l2 reg')
    parser.add_argument('--lr_actor', type=float, default=0.001, help='the learning rate of the actor')
    parser.add_argument('--lr_critic', type=float, default=0.001, help='the learning rate of the critic')
    parser.add_argument('--base_lr', type=float, default=0.001, help='the learning rate of the critic')
    parser.add_argument('--polyak', type=float, default=0.95, help='the average coefficient')
    parser.add_argument('--n_test_rollouts', type=int, default=10, help='the number of tests')
    parser.add_argument('--clip_range', type=float, default=5, help='the clip range')
    parser.add_argument('--state_norm', type=bool, default=True, help='is state norm or none')
    parser.add_argument('--demo_length', type=int, default=20, help='the demo length')
    parser.add_argument('--cuda', type=bool, default=True, help='if use gpu do the acceleration')
    parser.add_argument('--gpu_id',
                        type=float, default=gpu_id, help='gpu id')
    parser.add_argument('--render', type=bool, default=False, help='if render')
    parser.add_argument('--sess_opt', type=float, default=0.1, help='the Memory-Usage rate of GPU')
    parser.add_argument('--num_rollouts_per_mpi', type=int, default=2, help='the rollouts per mpi')
    parser.add_argument('--her', type=bool,
                        default=True, help='is HER True or False')
    parser.add_argument('--per', type=bool,
                        default=False, help='is PER True or False')

    parser.add_argument('--exp_name', type=str,
                        default='HER')
    parser.add_argument('--output_dir', type=str,
                        default='D2SSR_1obj_ep150_opt50_gd1_dense2_d2ssr_un40_exps')

    args = parser.parse_args()

    return args
