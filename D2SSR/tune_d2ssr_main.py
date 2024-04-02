import base64
from copy import deepcopy
import cloudpickle
import numpy as np
import os
import os.path as osp
import string
import subprocess
from subprocess import CalledProcessError
import sys
from textwrap import dedent
import time
import zlib

# 导入待执行的函数
from spinup_utils.mpi_tools import mpi_fork
from D2SSR.d2ssr_train_torch import launch as d2ssr_launch
from D2SSR.torch_arguments import get_args

DIV_LINE_WIDTH = 80


def call_experiment(thunk, net, thunk_params_dict_list, args, cpu_num, **kwargs):
    """
        :params_dict thunk:待启动的函数
        :params_dict params_dict:批量参数名
        :params kwargs: 其他的一些没考虑到的参数~用处不大，没事儿最好别写这个,容易造成混乱~    
        正常的函数，传入参数之后，就会直接执行。
        但是通过这个神奇的lambda，就可以即把参数传进去，又不执行。返回出一个函数
        再次调用的时候，只需要将返回值，加上括号，即当一个无参数传入的函数执行就可以了。
    """
    def thunk_plus():
        # Fork into multiple processes
        mpi_fork(cpu_num)
        # Run thunk
        thunk(net, thunk_params_dict_list, args)
    # lambda封装会让tune_func.py中导入MPI模块报初始化错误。
    # thunk_plus = lambda: thunk(params_dict)
    # mpi_fork(len(params_dict))
    pickled_thunk = cloudpickle.dumps(thunk_plus)
    encoded_thunk = base64.b64encode(zlib.compress(pickled_thunk)).decode('utf-8')
    # 默认mpi_fork函数和run_entrypoint.py是在同一个文件夹spinup_utils，因此获取mpi的绝对路径
    # 如果不行的话，自己添加entrypoint的绝对路径就行
    base_path = mpi_fork.__code__.co_filename
    run_entrypoint_path = base_path.replace(base_path.split('/')[-1], '')
    entrypoint = osp.join(run_entrypoint_path, 'run_entrypoint.py')
    # entrypoint = osp.join(osp.abspath(osp.dirname(__file__)), 'run_entrypoint.py')
    
    # subprocess的输入就是一个字符串列表，正常在命令行，该怎么输入，这个就该怎么写。
    cmd = [sys.executable if sys.executable else 'python', entrypoint, encoded_thunk]
    print("tune_exps_pid:", os.getpid())
    try:
        subprocess.check_call(cmd, env=os.environ)
    except CalledProcessError:
        err_msg = '\n'*3 + '='*DIV_LINE_WIDTH + '\n' + dedent("""
            Check the traceback above to see what actually went wrong. 
            """) + '='*DIV_LINE_WIDTH + '\n'*3
        print("err_msg", err_msg)
        raise


if __name__ == '__main__':
    import time
    # time.sleep(60*60*2)
    """
    如果有多张显卡的话，可以把备选显卡都放到下面的列表中，
    然后根据params_dict的实验参数组合数，为每个显卡分配进程数。
    依次最后设置gpu_id，从0到最大序号，依次启动进程。
    """
    gpus = [0, 1, 2, 3]
    cpu_num = 5
    gpu_id = 3

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[gpu_id])

    params_dict = {
        "sd": [1000, 2000, 3000, 4000, 5000],
        "un": [40, 80],
        # gcrl, noise, single
        'nr': [0.00], # reward noise size, m.
        'd2s': [50], # d2s: 奖励切换轮次        
        'exp': [0], # exp 0: d2sr, 1: d2s_nb, 2: d2s_nn, 3:d2s_nbn,        
        'drt': [2], # dense reward type: 1: -d1, 2:-d1-d2, 3: 1-tanh(d1)-tanh(d2) 4: 1-d1-d2
        'srt': [0], # sparse reward type: 0 ({-1, 0}), 1 ({0, 1})
        'gd': [1], # goal distribution: 1 (multi goal task), 5 (single goal task).
        'rf': [1e6],
        'bs': [2048], # batch size
        'rd': [0.2],  # random noise
        're': [0],  # render flag
        'gamma': [0.98], 
        'env': ['FetchPickAndPlace-v1', 'FetchPush-v1'],
    }
    args = get_args()
    from D2SSR.td3_per_her import TD3Torch

    """
        done = 0: ours, only done = 1 if ag_index = dg.
        done = 1: all_done, done = 1 if r = rp
        done = 2: no_done, done = 0
    """
    # mpi_fork(cpu_num)
    import itertools
    # 将字典变为排列组合列表
    params_list = [list(value) for value in itertools.product(*params_dict.values())]
    # 将列表列表变为单个文件的字典列表
    params_dict_list = [{key: cur_param.pop(0) for key, value in params_dict.items()} for cur_param in params_list]
    print(params_dict_list)
    print("num_exps:", len(params_dict_list),
          "cycle_num:", len(params_dict_list) // cpu_num)
    # input("sure continue ?")
    # 每次传入cpu_num数个字典。
    batch_count = 0
    for i in range(0, len(params_dict_list), cpu_num):
        cur_params_dict_list = params_dict_list[i:i+cpu_num]
        if batch_count % len(gpus) == gpu_id:
            for d in cur_params_dict_list:
                for key, value in d.items():
                    if key in ['sd', 'd2s', 'env']:
                        print("key, value", key, value, end='\t')
                print("")
            # print("cur_dict:", cur_params_dict_list)
            print("i:", i, "batch_count:", batch_count,
                  "gpus:", gpus, "gpu_id:", gpu_id)
            call_experiment(thunk=d2ssr_launch,
                            net=TD3Torch,
                            thunk_params_dict_list=cur_params_dict_list,
                            args=args,
                            cpu_num=cpu_num,
                            )

        batch_count += 1

