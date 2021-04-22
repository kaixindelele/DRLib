import base64
from copy import deepcopy
import cloudpickle
import numpy as np
import os
import os.path as osp
import psutil
import string
import subprocess
from subprocess import CalledProcessError
import sys
from textwrap import dedent
import time
import zlib

# 导入待执行的函数
from spinup_utils.mpi_tools import mpi_fork
from tune_func import func


DIV_LINE_WIDTH = 80


def call_experiment(thunk, thunk_params_dict_list, args, cpu_num, **kwargs):
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
        thunk(thunk_params_dict_list)
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
        print(err_msg)
        raise


if __name__ == '__main__':    
    cpu_num = 2
    params_dict = {
        'lr': [2, 3, 4],
        "batch": [10, 20, 30],
        "epoch": [9, 8, 7],
    }

    import itertools
    # 将字典变为排列组合列表
    params_list = [list(value) for value in itertools.product(*params_dict.values())]        
    # 将列表列表变为单个文件的字典列表
    params_dict_list = [{key: cur_param.pop(0) for key, value in params_dict.items()} for cur_param in params_list]
    print(params_dict_list)
    # 每次传入cpu_num数个字典。
    for i in range(0, len(params_dict_list), cpu_num):
        cur_params_dict_list = params_dict_list[i:i+cpu_num]
        print("cur_params_dict_list:", cur_params_dict_list)
        call_experiment(thunk=func,
                        thunk_params_dict_list=cur_params_dict_list,
                        args=args,
                        cpu_num=cpu_num)




