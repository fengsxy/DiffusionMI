import bmi
import numpy as np
from typing import Dict, List
import json
from datetime import datetime
import os
import argparse

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
# Importing all estimators
import bmi.benchmark
from src.estimators.neural import (
    CPCEstimator, DIMEEstimator, DoEEstimator, MINDEstimator,
    MINDEEstimator, MINEEstimator, NJEEEstimator, NWJEstimator, SMILEEstimator
)

def load_existing_results(filename: str) -> List[Dict]:
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return []

def append_results(new_results: List[Dict], filename: str):
    existing_results = load_existing_results(filename)
    existing_results.extend(new_results)
    with open(filename, 'w') as f:
        json.dump(existing_results, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run HighMutualInformation estimation benchmark')
    parser.add_argument('--task_start', type=int, default=0, help='Index of the first task to evaluate')
    parser.add_argument('--task_end', type=int, default=100, help='Index of the last task to evaluate')
    parser.add_argument('--max_steps', type=int, default=20000)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mi_estimation_interval', type=int, default=1000)
    parser.add_argument('--result_filename', type=str, default="result.json")
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--training_num', type=int, default=100000)
    parser.add_argument('--testing_num', type=int, default=10000)

    

    args = parser.parse_args()
    task_start = args.task_start
    task_end = args.task_end

    #strength = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
    strength = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    #estimator_list = [ DIMEEstimator, SMILEEstimator,MINDEEstimator,MINDEstimator,MINEEstimator,CPCEstimator]
    estimator_list = [MINDEstimator, MINEEstimator, MINDEEstimator]
    result_dict = []
    max_epochs = args.max_epochs
    seed = args.seed
    learning_rate = args.learning_rate
    mi_estimation_interval = args.mi_estimation_interval
    batch_size = args.batch_size
    result_filename = f"results/{args.result_filename}_seed_{seed}_batch_{batch_size}_max_epochs_{max_epochs}_mi_interval_{mi_estimation_interval}.json"

    for est in estimator_list:
        for s in strength[task_start:task_end]:
            task = bmi.benchmark.tasks.task_multinormal_sparse(dim_x=3, dim_y=3, strength=s)
            half_cube_task = bmi.benchmark.tasks.transform_half_cube_task(task)
            spiral_cube_task = bmi.benchmark.tasks.transform_spiral_task(task)
            task_list = [task, half_cube_task, spiral_cube_task]
            for task in task_list:
                task_name = task.name
                
                if task.dim_x <= 10:
                    LR = 1e-3
                    batch_size = 128
                    dim = 128
                elif task.dim_x <= 50:
                    batch_size = 256
                    LR = 2e-3
                    dim = 128
                else:
                    LR = 2e-3
                    batch_size = 256
                    dim = 256
                HIDDEN_layer = (dim, dim, 8)

                estimator = est(
                    x_shape=(task.dim_x,), 
                    y_shape=(task.dim_y,), 
                    learning_rate=learning_rate, 
                    batch_size=batch_size,
                    max_epcohs=max_epochs,
                    seed=seed, 
                    task_name=f"{task_name}_mutual_information_{task.mutual_information}",
                    mi_estimation_interval=mi_estimation_interval,
                    task_gt=task.mutual_information
                )
                


                X, Y = task.sample(args.training_num, seed=args.seed)
                X, Y = X.__array__(), Y.__array__()
                estimator.fit(X, Y)

                X, Y = task.sample(args.testing_num, seed=args.seed)
                X, Y = X.__array__(), Y.__array__()
                mi_estimate = estimator.estimate(X, Y)

                result_dict.append( {
                    "task": task.name,
                    "gt_mi": task.mutual_information,
                    "mi_estimate": mi_estimate[0] if isinstance(mi_estimate, tuple) else mi_estimate,
                    "mi_estimate_o": mi_estimate[1] if isinstance(mi_estimate, tuple) else None,
                    "learning_rate": args.learning_rate,
                    "estimator": est.__class__.__name__,
                    "seed": seed,
                    "batch_size": batch_size,
                    "train_sample_num": args.train_sample_num,
                    "test_sample_num": args.test_sample_num,
                    "max_epochs": max_epochs,
                    "max_steps": None,
                    "hidden_dim": 64 if task.dim_x <= 10 else 128 if task.dim_x <= 50 else 256,
                    "time_emb_size": 64 if task.dim_x <= 10 else 128 if task.dim_x <= 50 else 256,
                    "n_layers": None
                    })
                def append_results(result_dict, filename):
                    if os.path.exists(filename):
                        with open(filename, 'r') as f:
                            results = json.load(f)
                    else:
                        results = []
                    results.append(result_dict)
                    with open(filename, 'w') as f:
                        json.dump(results, f, indent=2)
                append_results(result_dict, 'results.json')