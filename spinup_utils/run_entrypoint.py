"""
如果用mpi_fork的话，就要多次启动当前脚本。
如果不进行隔离的话，在run_utils.py中的for var in vars中执行ppo.
执行到ppo中的mpi_fork(num_cpu)这句话时，会

"""


import zlib
import pickle
import base64
import time
from spinup_utils.mpi_tools import proc_id


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # 为什么加了这个，就能直接获取这个变量？
    # 当执行python run_entrypoint.py encoded_thunk 时，
    parser.add_argument('encoded_thunk')
    args = parser.parse_args()
    # print("thunk.args:", args)
    # input(("args"))
    # pickle.loads是读取函数
    thunk = pickle.loads(zlib.decompress(base64.b64decode(args.encoded_thunk)))
    # print("thunk:", thunk)
    # print("entry_point_proc_id:", proc_id())
    # time.sleep(1)
    thunk()


