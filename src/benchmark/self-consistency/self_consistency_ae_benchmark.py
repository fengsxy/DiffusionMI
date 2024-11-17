import torch
import torchvision
from torchvision import transforms
from typing import Tuple, Literal
from huggingface_ae import AE
import sys
import os
import json
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.estimators.neural import (
    CPCEstimator, DIMEEstimator, DoEEstimator, MINDEstimator,
    MINDEEstimator, MINEEstimator, NJEEEstimator, NWJEstimator, SMILEEstimator
)

BMI_ESTIMATORS = [
     DIMEEstimator, DoEEstimator, MINDEstimator,
    MINDEEstimator, MINEEstimator, NJEEEstimator, NWJEstimator, SMILEEstimator
]

DatasetType = Literal['mnist', 'fashionmnist', 'cifar10']
TaskType = Literal['baseline', 'data_processing', 'additivity']

class Self_Consistency_Benchmark:
    def __init__(self, task_type: TaskType, dataset: DatasetType, rows: int, k: int = 4):
        self.task_type = task_type
        self.dataset = dataset
        self.rows = rows
        self.k = k

        if dataset == 'mnist' or dataset == 'fashionmnist':
            self.dim_x = (1, 28, 28)
            self.dim_y = (1, 28, 28)
            self.model = AE.from_pretrained(f"liddlefish/mnist_auto_encoder_crop_{rows}")
        elif dataset == 'cifar10':
            self.dim_x = (3, 32, 32)
            self.dim_y = (3, 32, 32)
            raise NotImplementedError("CIFAR10 autoencoder not implemented")

        self.name = f"{task_type}_{dataset}_{rows}"
        if task_type == 'data_processing':
            self.name += f"_{k}"

    def sample(self, n: int, seed: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seed is not None:
            torch.manual_seed(seed)
        
        train_dataset = self.load_dataset()
        
        sampler = torch.utils.data.RandomSampler(train_dataset, num_samples=n, replacement=True)
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=n, sampler=sampler)
        if self.task_type == 'baseline':
            X, _ = next(iter(dataloader))
            X, Y = self.transform(X)
        elif self.task_type == 'data_processing':
            X, _ = next(iter(dataloader))
            X, Y = self.transform(X)
        elif self.task_type == 'additivity':
            X_1, _ = next(iter(dataloader))
            X_2, _ = next(iter(dataloader))
            X, Y = self.transform(X_1, X_2)

        return X, Y

    def load_dataset(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) if self.dataset != 'cifar10' else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        if self.dataset == 'mnist':
            return torchvision.datasets.MNIST('data', train=True, download=True, transform=transform)
        elif self.dataset == 'cifar10':
            return torchvision.datasets.CIFAR10('data', train=True, download=True, transform=transform)
        elif self.dataset == 'fashionmnist':
            return torchvision.datasets.FashionMNIST('data', train=True, download=True, transform=transform)

    def standardize(self, latent: tuple) -> torch.Tensor:
        latent = latent[0]
        mean = latent.mean(dim=0)
        std = latent.std(dim=0)
        std[std == 0] = 1  # Avoid division by zero
        return (latent - mean) / std
    
    def transform(self, X_1: torch.Tensor, X_2: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.task_type == 'baseline':
            Y = X_1.clone()
            Y[:, :, self.rows:, :] = 0.0
            self.model.eval()
            with torch.no_grad():
                X_latent = self.model.encode(X_1)
                Y_latent = self.model.encode(Y)
                X_latent = self.standardize(X_latent)
                Y_latent = self.standardize(Y_latent)
            return X_latent, Y_latent
        elif self.task_type == 'data_processing':
            Y1 = X_1.clone()
            Y1[:, :, self.rows:, :] = 0.0
            Y2 = X_1.clone()
            Y2[:, :, self.rows-self.k:, :] = 0.0
            with torch.no_grad():
                X_latent = self.model.encode(X_1)
                Y1_latent = self.model.encode(Y1)
                Y2_latent = self.model.encode(Y2)
                X_latent = self.standardize(X_latent)
                Y1_latent = self.standardize(Y1_latent)
                Y2_latent = self.standardize(Y2_latent)
            X_latent = torch.cat((X_latent, X_latent), dim=1)
            Y_latent = torch.cat((Y1_latent, Y2_latent), dim=1)
            return X_latent, Y_latent
        elif self.task_type == 'additivity':
            with torch.no_grad():
                Y1 = X_1.clone()
                Y2 = X_2.clone()
                Y1[:, :, self.rows:, :] = 0.0
                Y2[:, :, self.rows:, :] = 0.0
                X_1_latent = self.model.encode(X_1)
                X_2_latent = self.model.encode(X_2)
                Y_1_latent = self.model.encode(Y1)
                Y_2_latent = self.model.encode(Y2)
                X_1_latent = self.standardize(X_1_latent)
                X_2_latent = self.standardize(X_2_latent)
                Y_1_latent = self.standardize(Y_1_latent)
                Y_2_latent = self.standardize(Y_2_latent)
            X_latent = torch.cat((X_1_latent, X_2_latent), dim=1)
            Y_latent = torch.cat((Y_1_latent, Y_2_latent), dim=1)
            return X_latent, Y_latent

    @property
    def mutual_information(self) -> float:
        if self.task_type == 'baseline':
            return self.rows / 28  # Adjusted MI
        elif self.task_type == 'data_processing':
            return 1.0  # Ideal value
        elif self.task_type == 'additivity':
            return 2.0  # Ideal value

def append_to_json_file(file_path, new_data):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []
    
    data.append(new_data)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == '__main__':
    Path("result").mkdir(exist_ok=True)
    import argparse
    parser = argparse.ArgumentParser(description='Run HighMutualInformation estimation benchmark')
    parser.add_argument('--task_start', type=int, default=0, help='Index of the first task to evaluate')
    parser.add_argument('--task_end', type=int, default=100, help='Index of the last task to evaluate')
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--max_epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mi_estimation_interval', type=int, default=1000)
    parser.add_argument('--result_filename', type=str, default="result.json")
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--training_num', type=int, default=100000)
    parser.add_argument('--testing_num', type=int, default=10000)
    parser.add_argument('--hidden_num', type=tuple, default=())
    
    args = parser.parse_args()

    row_list = [0, 5, 10, 15, 20, 25, 28]
    task_type_list = ['baseline', 'data_processing', 'additivity']
    test_estimator_list = [MINDEstimator]



    for estimator in test_estimator_list:
        for task_type in task_type_list:
            file_path = f"result/{estimator.__name__}_{task_type}.json"
            for row in row_list[args.task_start:args.task_end]:
                if row < 5 and task_type != 'data_processing':
                    continue
                baseline_task = Self_Consistency_Benchmark(task_type=task_type, dataset='mnist', rows=row, k=4)
                X_latent, Y_latent = baseline_task.sample(args.training_num)
                task_x_dim = X_latent.shape[1]
                task_y_dim = Y_latent.shape[1]
                est = estimator(
                    x_shape=(task_x_dim,), 
                    y_shape=(task_y_dim,), 
                    max_n_steps=args.max_steps,
                    max_epochs=args.max_epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    seed=args.seed,
                    mi_estimation_interval=args.mi_estimation_interval
                )
                estimator_name = est.__class__.__name__
                est.fit(X_latent, Y_latent)
                X_test, Y_test = baseline_task.sample(args.testing_num, seed=args.seed)
                mi_estimate = est.estimate(X_test, Y_test)
                
                result = {
                    "row": row,
                    "mi_estimate": mi_estimate[0] if isinstance(mi_estimate, tuple) else mi_estimate,
                    "mi_estimate_o": mi_estimate[1] if isinstance(mi_estimate, tuple) else None,
                    "estimator": estimator_name,
                    "task_type": task_type,
                    "task": baseline_task.name,
                    "learning_rate": args.learning_rate,
                    "seed": args.seed,
                    "batch_size": args.batch_size,
                    "train_sample_num": args.training_num,
                    "test_sample_num": args.testing_num,
                    "max_epochs": args.max_epochs,
                    "max_steps": args.max_steps,
                    "mi_estimation_interval": args.mi_estimation_interval,
                    "hidden_dim": 64 if task_x_dim <= 10 else 128 if task_x_dim <= 50 else 256,
                    "time_emb_size": 64 if task_x_dim <= 10 else 128 if task_x_dim <= 50 else 256,
                    "n_layers": None
                }
                
                append_to_json_file(file_path, result)
                print(f"Result for row {row} saved to {file_path}")