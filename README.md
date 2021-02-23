# DRLib
My DRL library with tensorflow1.14 and pytorch, add HER and PER, core codes based on https://github.com/openai/spinningup

## Installation
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
    ```bash 
    refer to https://blog.csdn.net/hehedadaq/article/details/110423154
    ```

