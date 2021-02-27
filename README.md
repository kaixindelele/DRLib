# DRLib：A concise deep reinforcement learning library which integrats amost all of off policy RL algos with HER and PER.
A concise deep reinforcement learning library which integrats amost all of off policy RL algos with HER and PER. The library is written based on the code in https://github.com/openai/spinningup, and can be achieved with tensorflow or pytorch.
Compared with spinning up, the multi-process and experimental grid wrapper have been deleted for easy application. Besides, the code in our library is convenient to debug with pycharm~

## 项目特点：

1. tf1和pytorch两个版本的算法，前者快，后者新，任君选择；

2. 在spinup的基础上，封装了DDPG, TD3, SAC等主流强化算法，相比原来的函数形式的封装，调用更方便，且**加了pytorch的GPU调用**；

3. **添加了HER和PER功能**，非常适合做机器人相关任务的同学们；

4. 去除了自动调参（ExperimentGrid）和多进程（MPI_fork）部分，适合新手在pycharm中debug，前者直接跑经常会报错~
等我熟练了这两个，我再加上去，并附上详细教程；

5. 最后，全网最详细的环境配置教程！**亲测两个小时内，从零配置完全套环境！**

6. **求三连，不行求个star！**

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
    
    If installation of mpi4py fails, try the following command(Only this one can be installed successfully!):
    ```bash
    conda install mpi4py
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
1. modify params in arguments.py, choose env, RL-algorithm, use PER and HER or not, gpu-id, and so on.
2. run with train_tf.py or train_torch.py
    ```bash 
    python train_tf.py
    ```
3. exp results to local:https://blog.csdn.net/hehedadaq/article/details/114045615
4. plot results:https://blog.csdn.net/hehedadaq/article/details/114044217
    
## 3. File tree and introduction:

<p float="middle">
  <img src="https://img-blog.csdnimg.cn/20210224181411670.png" />
</p>



```bash
.
├── algos
│   ├── pytorch
│   │   ├── ddpg_sp
│   │   │   ├── core.py-------------It's copied directly from spinup, and modified some details.
│   │   │   ├── ddpg_per_her.py-----inherits from offPolicy.baseOffPolicy, where one can choose whether or not HER and PER
│   │   │   ├── ddpg.py-------------It's copied directly from spinup
│   │   │   ├── __init__.py
│   │   ├── __init__.py
│   │   ├── offPolicy
│   │   │   ├── baseOffPolicy.py----baseOffPolicy, DDPG/TD3/SAC and so on.
│   │   │   ├── norm.py-------------state normalizer, update mean/std with training process.
│   │   ├── sac_auto
│   │   ├── sac_sp
│   │   │   ├── core.py-------------likely as before.
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
│       │   ├── DDPG_class.py------------It's copied directly from spinup, and wrap algorithm from function to class.
│       │   ├── DDPG_per_class.py--------Add PER.
│       │   ├── DDPG_per_her_class.py----DDPG with HER and PER without inheriting from offPolicy.
│       │   ├── DDPG_per_her.py----------Add HER and PER.
│       │   ├── DDPG_sp.py---------------It's copied directly from spinup, and modified some details.
│       │   ├── __init__.py
│       ├── __init__.py
│       ├── offPolicy
│       │   ├── baseOffPolicy.py
│       │   ├── core.py
│       │   ├── norm.py
│       ├── sac_auto--------------------SAC with auto adjust alpha parameter version.
│       │   ├── core.py
│       │   ├── __init__.py
│       │   ├── sac_auto_class.py
│       │   ├── sac_auto_per_class.py
│       │   └── sac_auto_per_her.py
│       ├── sac_sp--------------------SAC with alpha=0.2 version.
│       │   ├── core.py
│       │   ├── __init__.py
│       │   ├── SAC_class.py
│       │   ├── SAC_per_class.py
│       │   ├── SAC_per_her.py
│       │   ├── SAC_sp.py
│       └── td3_sp
│           ├── core.py
│           ├── __init__.py
│           ├── TD3_class.py
│           ├── TD3_per_class.py
│           ├── TD3_per_her_class.py
│           ├── TD3_per_her.py
│           ├── TD3_sp.py
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
│   ├── per_memory.py--------------mofan version
│   ├── simple_memory.py-----------mofan version
│   ├── sp_memory.py---------------spinningup tf1 version, simple uniform buffer memory class.
│   ├── sp_memory_torch.py---------spinningup torch-gpu version, simple uniform buffer memory class.
│   ├── sp_per_memory.py-----------spinningup tf1 version, PER buffer memory class.
│   └── sp_per_memory_torch.py
├── pip_requirement.txt------------pip install requirement, exclude mujoco-py,gym,tf,torch.
├── spinup_utils-------------------some utils from spinningup, about ploting results, logging, and so on.
│   ├── delete_no_checkpoint.py----delete the folder where the experiment did not complete.
│   ├── __init__.py
│   ├── logx.py
│   ├── mpi_tf.py
│   ├── mpi_tools.py
│   ├── plot.py
│   ├── print_logger.py------------save the information printed by the terminal to the local log file。
│   ├── run_utils.py---------------now I haven't used it. I have to learn how to multi-process.
│   ├── serialization_utils.py
│   └── user_config.py
├── train_tf1.py--------------main.py for tf1
└── train_torch.py------------main.py for torch


```


## 4. HER introduction:

the achievement of HER is based on the following code :

1. It can be converged, but this code is too difficult. https://github.com/openai/baselines

2. It can also converged, but only for DDPG-torch-cpu. https://github.com/sush1996/DDPG_Fetch

3. It can not be converged, but this code is simpler. https://github.com/Stable-Baselines-Team/stable-baselines

- paper: https://arxiv.org/pdf/1709.10089.pdf


### 4.1. My understanding and video:

>种瓜得豆来解释her:
第一步在春天（state），种瓜（origin-goal）得豆，通过HER，把目标换成种豆，按照之前的操作，可以学会在春天种豆得豆；
第二步种米得瓜，学会种瓜得瓜；
即只要是智能体中间经历过的状态，都可以当做它的目标，进行学会。
即如果智能体能遍历所有的状态空间，那么它就可以学会达到整个状态空间。

https://www.bilibili.com/video/BV1BA411x7Wm

### 4.2. Key tricks for HER:

1. state-normalize: success rate from 0 to 1 for FetchPush-v1 task.
2. Q-clip: success rate from 0.5 to 0.7 for FetchPickAndPlace-v1 task.
3. action_l2: little effect for Push task.

### 4.3. Performance about HER-DDPG with FetchPush-v1:

<p float="middle">
  <img src="https://img-blog.csdnimg.cn/2021022323234470.png" />
</p>

## 5. PER introduction:

refer to:[off-policy全系列（DDPG-TD3-SAC-SAC-auto）+优先经验回放PER-代码-实验结果分析](https://blog.csdn.net/hehedadaq/article/details/111600080)

## 6. Summary：

这个库我封装了好久，整个代码库简洁、方便、功能比较齐全，在环境配置这块几乎是手把手教程，希望能给大家节省一些时间~

从零开始配置，不到两小时，从下载代码库，到配置环境，到在自己的环境中跑通，全流程非常流畅。

### 6.1. 下一步添加的功能：

1. PPO的封装；

2. DQN的封装；

3. 多进程的封装；

4. ExperimentGrid的封装；


## 7. Contact：

深度强化学习-DRL：799378128

欢迎关注知乎帐号：[未入门的炼丹学徒](https://www.zhihu.com/people/heda-he-28)

CSDN帐号：[https://blog.csdn.net/hehedadaq](https://blog.csdn.net/hehedadaq)
