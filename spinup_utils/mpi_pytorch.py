import multiprocessing
import numpy as np
import os
import torch
from mpi4py import MPI
from spinup_utils.mpi_tools import broadcast, mpi_avg, num_procs, proc_id


def setup_pytorch_for_mpi():
    """
    Avoid slowdowns caused by each separate process's PyTorch using
    more than its fair share of CPU resources.
    """
    print('Proc %d: Reporting original number of Torch threads as %d.'%(proc_id(), torch.get_num_threads()), flush=True)
    if torch.get_num_threads() == 1:
        return
    fair_num_threads = max(int(torch.get_num_threads() / num_procs()), 1)
    torch.set_num_threads(fair_num_threads)
    print('Proc %d: Reporting new number of Torch threads as %d.'%(proc_id(), torch.get_num_threads()), flush=True)


def mpi_avg_grads(module, index=None):
    """ Average contents of gradient buffers across MPI processes. """
    if num_procs() == 1:
        return
    count = 0
    for p in module.parameters():
        p_grad_numpy = p.grad.numpy()   # numpy view of tensor data
        # if count == 0 and index is not None:
        #     try:
        #         print("in_bef_MPI_rank:", MPI.COMM_WORLD.Get_rank(),
        #               "index:", index,
        #               "p_grad_numpy:", p_grad_numpy[0][0])
        #     except:
        #         pass
        avg_p_grad = mpi_avg(p.grad)
        # type(avg_p_grad) is numpy.ndarray
        # if count == 0 and index is not None:
        #     try:
        #         print("in_aft_MPI_rank:", MPI.COMM_WORLD.Get_rank(),
        #               "index:", index,
        #               "avg_p_grad:", avg_p_grad[0][0])
        #         count = 1
        #     except:
        #         pass
        p_grad_numpy[:] = avg_p_grad[:]


def sync_params(module):
    """ Sync all parameters of module across all MPI processes. """
    if num_procs()==1:
        return
    for p in module.parameters():
        p_numpy = p.data.numpy()
        broadcast(p_numpy)

