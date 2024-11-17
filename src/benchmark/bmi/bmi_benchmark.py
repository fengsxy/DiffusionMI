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
# 导入所有估计器
from src.estimators.neural import (
    CPCEstimator, DIMEEstimator, DoEEstimator, MINDEstimator,
    MINDEEstimator, MINEEstimator, NJEEEstimator, NWJEstimator, SMILEEstimator
)


BMI_ESTIMATORS = [
     DIMEEstimator, DoEEstimator, MINDEstimator,
    MINDEEstimator, MINEEstimator, NJEEEstimator, NWJEstimator, SMILEEstimator
]



def evaluate_estimator(estimator_class, task, n_trials: int = 5, train_samples: int = 10000, test_samples: int = 1000, max_steps: int = 500, batch_size: int = 256) -> Dict:
    results = []

    for _ in range(n_trials):
        estimator = estimator_class(x_shape=(task.dim_x,), y_shape=(task.dim_y,), max_n_steps=max_steps, batch_size=batch_size)
        
        X_train, Y_train = task.sample(train_samples, seed=42)
        X_train = X_train.__array__()
        Y_train = Y_train.__array__()
        
        if hasattr(estimator, 'fit'):
            estimator.fit(X_train, Y_train)
        
        X_test, Y_test = task.sample(test_samples, seed=43)
        X_test = X_test.__array__()
        Y_test = Y_test.__array__()
        
        mi_estimate = estimator.estimate(X_test, Y_test)
        results.append(mi_estimate)

    mean_estimate = np.mean(results)
    std_estimate = np.std(results)
    true_mi = task.mutual_information
    absolute_error = np.abs(mean_estimate - true_mi)
    relative_error = absolute_error / true_mi if true_mi != 0 else np.inf

    return {
        'estimator': estimator_class.__name__,
        'task': task.name,
        'true_mi': true_mi,
        'mean_estimate': mean_estimate,
        'std_estimate': std_estimate,
        'absolute_error': absolute_error,
        'relative_error': relative_error
    }

def load_existing_results(filename: str) -> List[Dict]:
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return []

def save_results(results: List[Dict], filename: str):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def run_benchmark(tasks: List[str], estimators: List, skip_estimators: List[str] = None, n_trials: int = 5, train_samples: int = 10000, test_samples: int = 1000, max_steps: int = 500, batch_size: int = 256, result_file: str = None) -> List[Dict]:
    if result_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f'bmi_benchmark_results_{timestamp}.json'
    
    results = load_existing_results(result_file)
    completed_tasks = set((r['estimator'], r['task']) for r in results)

    if skip_estimators is None:
        skip_estimators = []

    for task_name in tasks:
        task = bmi.benchmark.BENCHMARK_TASKS[task_name]
        for estimator_class in estimators:
            estimator_name = estimator_class.__name__
            if estimator_name in skip_estimators:
                print(f"Skipping {estimator_name} as requested")
                continue

            if (estimator_name, task_name) in completed_tasks:
                print(f"Skipping {estimator_name} on {task_name}: Already evaluated")
                continue
            
            try:
                result = evaluate_estimator(estimator_class, task, n_trials, train_samples, test_samples, max_steps, batch_size)
                results.append(result)
                print(f"Evaluated {estimator_name} on {task_name}: Mean={result['mean_estimate']:.4f}, Std={result['std_estimate']:.4f}, Relative Error={result['relative_error']:.4f}")
            except Exception as e:
                print(f"Error evaluating {estimator_name} on {task_name}: {str(e)}")
                continue
            
            save_results(results, result_file)
            print(f"Results updated in {result_file}")

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run BMI estimation benchmark')
    parser.add_argument('--skip', nargs='+', help='List of BMI estimators to skip', default=[])
    parser.add_argument('--trials', type=int, default=5, help='Number of trials for each estimator')
    parser.add_argument('--train_samples', type=int, default=10000, help='Number of training samples for each trial')
    parser.add_argument('--test_samples', type=int, default=1000, help='Number of test samples for each trial')
    parser.add_argument('--max_steps', type=int, default=5000, help='Maximum number of steps for training')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=1)
    parser.add_argument('--max_steps', type=int, default=20000)
    parser.add_argument('--update_logsnr_loc_flag', type=bool, default=False)
    args = parser.parse_args()

    tasks_to_evaluate = list(bmi.benchmark.BENCHMARK_TASKS.keys())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f'bmi_benchmark_results_{timestamp}.json'
    
    task_list = list(bmi.benchmark.BENCHMARK_TASKS.keys())
    #set start and end with arguments
    import argparse
    parser = argparse.ArgumentParser()
    

    arg = parser.parse_args()
    start = arg.start
    end = arg.end
    for task_name in task_list[start:end]:
        for estimator_class in BMI_ESTIMATORS:
            task = bmi.benchmark.BENCHMARK_TASKS[task_name]
            train_sample_num = arg.training_num
            test_sample_num = arg.testing_num
            #max_steps = arg.max_steps
            batch_size = 256
            seed = 42
            mi_estimation_interval = 500
            update_logsnr_loc_flag = arg.update_logsnr_loc_flag
            

            X, Y = task.sample(train_sample_num, seed=seed)
            X_test, Y_test = task.sample(test_sample_num, seed=seed)
            X, Y = X.__array__(), Y.__array__()
            X_test, Y_test = X_test.__array__(), Y_test.__array__()
        

            if task.dim_x <= 5:
                max_epochs = 300
                lr = 1e-3
                batch_size = 128
            elif task.dim_x > 5:
                max_epochs = 500
                lr = 2e-3
                batch_size = 256

            est = estimator_class(
                x_shape=(task.dim_x,), 
                y_shape=(task.dim_y,), 
                learning_rate=lr, 
                batch_size=batch_size,
                max_epochs=max_epochs,
                seed=seed, 
                task_name=task_name,
                mi_estimation_interval=mi_estimation_interval,
                task_gt=task.mutual_information,
            )
            est.fit(X, Y)

            mi_estimate = est.estimate(X_test, Y_test)
            
            #save as json
            import json

            result_dict = {
                "task": task.name,
                "gt_mi": task.mutual_information,
                "mi_estimate": mi_estimate[0] if isinstance(mi_estimate, tuple) else mi_estimate,
                "mi_estimate_o": mi_estimate[1] if isinstance(mi_estimate, tuple) else None,
                "learning_rate": lr,
                "estimator": est.__class__.__name__,
                "seed": seed,
                "batch_size": batch_size,
                "train_sample_num": train_sample_num,
                "test_sample_num": test_sample_num,
                'log_snr_dynamic': update_logsnr_loc_flag,
                "max_epochs": max_epochs,
                "max_steps": None,
                "hidden_dim": 64 if task.dim_x <= 10 else 128 if task.dim_x <= 50 else 256,
                "time_emb_size": 64 if task.dim_x <= 10 else 128 if task.dim_x <= 50 else 256,
                "n_layers": None
                }
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