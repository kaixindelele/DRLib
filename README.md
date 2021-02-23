# DRLib：A concise deep reinforcement learning library, integrating HER and PER for almost off policy RL algos
My DRL library with tensorflow1.14 and pytorch, add HER and PER, core codes based on https://github.com/openai/spinningup

## 1. Installation
1. Clone the repo and cd into it:
    ```bash
    git clone https://github.com/kaixindelele/DRLib.git
    cd DRLib
    ```
2. Create anaconda DRLib_env env:
    ```bash
    conda create -n DRLib_env python=3.6.9
    source activate DRLib_env
    ```
3. Install pip_requirement.txt:
    ```bash
    pip install -r pip_requirement.txt
    ```
    
4. Install tensorflow-gpu=1.14.0
    ```bash 
    conda install tensorflow-gpu==1.14.0 # if you have a CUDA-compatible gpu and proper drivers
    ```
    
5. Install torch torchvision
    ```bash 
    # CUDA 9.2
    conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=9.2 -c pytorch

    # CUDA 10.1
    conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch

    # CUDA 10.2
    conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch

    # CPU Only
    conda install pytorch==1.6.0 torchvision==0.7.0 cpuonly -c pytorch
    
    # or pip install    
    pip --default-timeout=100 install torch -i  http://pypi.douban.com/simple  --trusted-host pypi.douban.com
    [pip install torch 在线安装!非离线!](https://blog.csdn.net/hehedadaq/article/details/111480313)
    ```
    
6. Install mujoco and mujoco-py
    ```bash 
    refer to: https://blog.csdn.net/hehedadaq/article/details/109012048
    ```
    
7. Install gym[all]
    ```python
    refer to https://blog.csdn.net/hehedadaq/article/details/110423154
    ```


## 2. Training models

- Example 1. SAC-tf1-HER-PER with FetchPush-v1:
    ```bash 
    python train_tf.py
    ```
    
## 3. File tree and introduction:
```bash
.
├── algos
│   ├── pytorch
│   │   ├── ddpg_sp
│   │   │   ├── core.py
│   │   │   ├── ddpg_per_her.py-----inherits from offPolicy.baseOffPolicy, can choose whether or not HER and PER
│   │   │   ├── ddpg.py-------------It's copied directly from spinup
│   │   │   ├── __init__.py
│   │   ├── __init__.py
│   │   ├── offPolicy
│   │   │   ├── baseOffPolicy.py----baseOffPolicy, can be used to DDPG/TD3/SAC and so on.
│   │   │   ├── norm.py
│   │   ├── sac_auto
│   │   ├── sac_sp
│   │   │   ├── core.py
│   │   │   ├── __init__.py
│   │   │   ├── sac_per_her.py
│   │   │   └── sac.py
│   │   └── td3_sp
│   │       ├── core.py
│   │       ├── __init__.py
│   │       ├── td3_gpu_class.py----td3_class modified from spinup
│   │       └── td3_per_her.py
│   └── tf1
│       ├── ddpg_sp
│       │   ├── core.py
│       │   ├── DDPG_class.py
│       │   ├── DDPG_per_class.py
│       │   ├── DDPG_per_her_class.py
│       │   ├── DDPG_per_her.py
│       │   ├── DDPG_sp.py
│       │   ├── __init__.py
│       ├── __init__.py
│       ├── offPolicy
│       │   ├── baseOffPolicy.py
│       │   ├── core.py
│       │   ├── norm.py
│       ├── sac_auto
│       │   ├── core.py
│       │   ├── __init__.py
│       │   ├── sac_auto_class.py
│       │   ├── sac_auto_per_class.py
│       │   └── sac_auto_per_her.py
│       ├── sac_sp
│       │   ├── core.py
│       │   ├── __init__.py
│       │   ├── SAC_class.py
│       │   ├── SAC_per_class.py
│       │   ├── SAC_per_her.py
│       │   ├── SAC_sp.py
│       │   └── test_gym_sac_sp_class.py
│       └── td3_sp
│           ├── core.py
│           ├── __init__.py
│           ├── TD3_cem_class.py
│           ├── TD3_cem_class_time_analysis.py
│           ├── TD3_class.py
│           ├── TD3_per_class.py
│           ├── TD3_per_class_time_analysis.py
│           ├── TD3_per_her_class.py
│           ├── TD3_per_her.py
│           ├── TD3_sp.py
│           └── time_wrap.py
├── arguments.py-----------------------hyperparams scripts
├── drlib_tree.txt
├── HER_DRLib_exps---------------------demo exp logs
│   ├── 2021-02-21_HER_TD3_FetchPush-v1
│   │   ├── 2021-02-21_18-26-08-HER_TD3_FetchPush-v1_s123
│   │   │   ├── checkpoint
│   │   │   ├── config.json
│   │   │   ├── params.data-00000-of-00001
│   │   │   ├── params.index
│   │   │   ├── progress.txt
│   │   │   └── Script_backup.py
├── memory
│   ├── __init__.py
│   ├── per_memory.py
│   ├── simple_memory.py
│   ├── sp_memory.py
│   ├── sp_memory_torch.py
│   ├── sp_per_memory.py
│   └── sp_per_memory_torch.py
├── pip_requirement.txt
├── spinup_utils
│   ├── delete_no_checkpoint.py
│   ├── __init__.py
│   ├── logx.py
│   ├── mpi_tf.py
│   ├── mpi_tools.py
│   ├── plot.py
│   ├── print_logger.py
│   ├── run_utils.py
│   ├── serialization_utils.py
│   └── user_config.py
├── train_tf1.py--------------main.py for tf1
└── train_torch.py------------main.py for torch


```


## 4. HER introduction:

Refer to these code bases:

1. It can be converged, but this code is too difficult. https://github.com/openai/baselines

2. It can also converged, but only for DDPG-torch-cpu. https://github.com/sush1996/DDPG_Fetch

3. It can not be converged, but this code is simpler. https://github.com/Stable-Baselines-Team/stable-baselines

- paper:

https://arxiv.org/pdf/1709.10089.pdf

### Key tricks for HER:

1. state-normalize: success rate from 0 to 1 for FetchPush-v1 task.
2. Q-clip: success rate from 0.5 to 0.7 for FetchPickAndPlace-v1 task.
3. action_l2: little effect for Push task.

### Performance about HER-DDPG with FetchPush-v1:
