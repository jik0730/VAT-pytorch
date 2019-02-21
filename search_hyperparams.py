# Base code is from https://github.com/cs230-stanford/cs230-code-examples
"""Peform hyperparemeters search"""

import argparse
import os
from subprocess import check_call
from multiprocessing import Process
import sys

import utils

PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument(
    '--parent_dir',
    default='experiments/epsilon',
    help='Directory containing params.json')
parser.add_argument(
    '--data_dir', default='data', help="Directory containing the dataset")


def launch_training_job(parent_dir, data_dir, job_name, params, gpu_num):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name
    Args:
        model_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    cmd = "CUDA_VISIBLE_DEVICES={gpu_num} {python} train_semi.py --model_dir={model_dir} --data_dir {data_dir}".format(
        gpu_num=gpu_num, python=PYTHON, model_dir=model_dir, data_dir=data_dir)
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Perform hypersearch over one parameter
    epsilons = [2, 2.5, 3, 3.5, 4, 4.5, 5]
    seeds = [1, 2, 3]

    proc_args = []
    for epsilon in epsilons:
        for seed in seeds:
            # Modify the relevant parameter in params
            params = utils.Params(json_path)
            params.epsilon = epsilon
            params.SEED = seed

            # Launch job (name has to be unique)
            job_name = "epsilon_{}_SEED_{}".format(epsilon, seed)
            proc_args.append(
                [args.parent_dir, args.data_dir, job_name, params])

    num_workers = 1
    num_proc_per_worker = 3
    max_proc = num_workers * num_proc_per_worker
    procs = []
    for count, proc_arg in enumerate(proc_args):
        gpu_num = count % num_workers
        proc = Process(target=launch_training_job, args=(*proc_arg, gpu_num))
        procs.append(proc)
        proc.start()

        if (count + 1) % max_proc == 0 or (count + 1) == len(proc_args):
            for proc in procs:
                proc.join()
            procs = []
