import os
import sys
import multiprocessing as mp
import shlex

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run import setup_arguments, main


def _worker(job_str: str):
    job_args_list = shlex.split(job_str)
    assert job_args_list[0] == 'JOB'
    del job_args_list[0]
    print(job_args_list)
    job_args = setup_arguments(args=job_args_list)
    main(job_args)


def main_njobs(job_specs, njobs: int, start_method: str = "spawn"):
    try:
        mp.set_start_method(start_method, force=True)
    except RuntimeError:
        pass

    with mp.Pool(processes=njobs) as pool:
        pool.starmap(_worker, [(spec,) for spec in job_specs])
