import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.estimators.neural._critic import MLP, ConvCritic
import bmi

class MINEEstimator:
    def __init__(
        self,
        batch_size=64,
        max_n_steps=2000,
        learning_rate=5e-5,
        hidden_layers=(100, 100),
        smoothing_alpha=0.01,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_ma_et=True,
        x_shape=None,
        y_shape=None
    ):
        self.batch_size = batch_size
        self.max_n_steps = max_n_steps
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers
        self.smoothing_alpha = smoothing_alpha
        self.device = device
        self.critic = None
        self.ma_et = 1.0
        self.use_ma_et = use_ma_et
        self.buffer = None
        self.momentum = 0.9
        self.x_shape = x_shape
        self.y_shape = y_shape

    def _create_critic(self, input_dim):
        return MLP(input_dim, self.hidden_layers).to(self.device)

    def _create_image_critic(self, x_channels, y_channels):
        return ConvCritic(x_channels, y_channels).to(self.device)

    @staticmethod
    def _mine_lower_bound(t, t_shuffle):
        max_t_shuffle = torch.max(t_shuffle)
        log_sum_exp_t_shuffle = torch.log(torch.mean(torch.exp(t_shuffle - max_t_shuffle))) + max_t_shuffle
        
        mi_lb = torch.mean(t) - log_sum_exp_t_shuffle
        R = torch.exp(-mi_lb)
        return mi_lb, R

    def fit(self, X: np.ndarray, Y: np.ndarray):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        Y = torch.tensor(Y, dtype=torch.float32).to(self.device)

        if self.x_shape is None:
            self.x_shape = X.shape[1:]
        if self.y_shape is None:
            self.y_shape = Y.shape[1:]

        if self.critic is None:
            input_dim = np.prod(self.x_shape) + np.prod(self.y_shape)
            self.critic = self._create_critic(input_dim)

        optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        steps = 0

        pbar = tqdm(total=self.max_n_steps, desc="Training MINE")
        while steps < self.max_n_steps:
            for x_batch, y_batch in dataloader:
                if steps >= self.max_n_steps:
                    break

                y_shuffle = y_batch[torch.randperm(y_batch.size(0))]
                t = self.critic(x_batch, y_batch)
                t_shuffle = self.critic(x_batch, y_shuffle)
                mi_estimate, R = self._mine_lower_bound(t, t_shuffle)

                loss = -mi_estimate
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                steps += 1
                pbar.update(1)
                pbar.set_postfix({'loss': loss.item()})

                if steps >= self.max_n_steps:
                    break

        pbar.close()

    def estimate(self, X: np.ndarray, Y: np.ndarray) -> float:
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        Y = torch.tensor(Y, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            y_shuffle = Y[torch.randperm(Y.size(0))]
            t = self.critic(X, Y)
            t_shuffle = self.critic(X, y_shuffle)
            mi_estimate, R = self._mine_lower_bound(t, t_shuffle)

        print(f"Final estimate - MI: {mi_estimate.item()}, R: {R.item()}")
        return mi_estimate.item()

    def fit_image(self, X: np.ndarray, Y: np.ndarray):
        if X.ndim == 3:
            X = X.unsqueeze(1) 
        if Y.ndim == 3:
            Y = Y.unsqueeze(1)

        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        Y = torch.tensor(Y, dtype=torch.float32).to(self.device)

        if self.x_shape is None:
            self.x_shape = X.shape[1:]
        if self.y_shape is None:
            self.y_shape = Y.shape[1:]

        if self.critic is None:
            self.critic = self._create_image_critic(X.shape[1], Y.shape[1])

        optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        steps = 0

        pbar = tqdm(total=self.max_n_steps, desc="Training MINE for image data")
        while steps < self.max_n_steps:
            for x_batch, y_batch in dataloader:
                if steps >= self.max_n_steps:
                    break

                y_shuffle = y_batch[torch.randperm(y_batch.size(0))]
                t = self.critic(x_batch, y_batch)
                t_shuffle = self.critic(x_batch, y_shuffle)
                mi_estimate, R = self._mine_lower_bound(t, t_shuffle)

                loss = -mi_estimate
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                steps += 1
                pbar.update(1)
                pbar.set_postfix({'loss': loss.item()})

                if steps >= self.max_n_steps:
                    break

        pbar.close()

    def estimate_image(self, X: np.ndarray, Y: np.ndarray) -> float:
        if X.ndim == 3:
            X = X.unsqueeze(1) 
        if Y.ndim == 3:
            Y = Y.unsqueeze(1)

        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        Y = torch.tensor(Y, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            y_shuffle = Y[torch.randperm(Y.size(0))]
            t = self.critic(X, Y)
            t_shuffle = self.critic(X, y_shuffle)
            mi_estimate, R = self._mine_lower_bound(t, t_shuffle)

        print(f"Final estimate for image data - MI: {mi_estimate.item()}, R: {R.item()}")
        return mi_estimate.item()

if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../benchmark/self-consistency')))
    from self_consistency_benchmark import Self_Consistency_Benchmark

    # Test on normal data
    task = bmi.benchmark.BENCHMARK_TASKS['1v1-normal-0.75']
    print(f"Task: {task.name}")
    print(f"Task {task.name} with dimensions {task.dim_x} and {task.dim_y}")
    print(f"Ground truth mutual information: {task.mutual_information:.2f}")

    X, Y = task.sample(10000, seed=42)
    X = X.__array__()
    Y = Y.__array__()
    mine = MINEEstimator(use_ma_et=True, max_n_steps=3000, batch_size=256)
    
    mine.fit(X, X)
    mine.estimate(X, X)


    X_test, Y_test = task.sample(1000, seed=41)
    X_test = X_test.__array__()
    Y_test = Y_test.__array__()
    mi_estimate = mine.estimate(X_test, Y_test)
    print(f"MINE estimate for normal data: {mi_estimate:.2f}")
    print(f"Absolute error: {abs(mi_estimate - task.mutual_information):.2f}")

    # Test on image data
    mine_image = MINEEstimator(max_n_steps=100, use_ma_et=True)
    baseline_task_7 = Self_Consistency_Benchmark(task_type='baseline', dataset='mnist', rows=7)
    
    X_7, Y_7 = baseline_task_7.sample(10000, seed=42)
    print(X_7[0])
    mine_image.fit_image(X_7, Y_7)

    X_test_7, Y_test_7 = baseline_task_7.sample(64, seed=42)
    mi_estimate_image_7 = mine_image.estimate_image(X_test_7, Y_test_7)

    mutual_information_7 = baseline_task_7.mutual_information

    baseline_task_28 = Self_Consistency_Benchmark(task_type='baseline', dataset='mnist', rows=28)
    X_28, Y_28 = baseline_task_28.sample(10000, seed=42)
    mine_image.fit_image(X_28, Y_28)

    X_test_28, Y_test_28 = baseline_task_28.sample(64, seed=42)
    mi_estimate_image_28 = mine_image.estimate_image(X_test_28, Y_test_28)

    print(f"MINE estimate for 7x7 image data: {mi_estimate_image_7:.2f}")
    print(f"Ground truth mutual information (7x7): {mutual_information_7:.2f}")
    print(f"Absolute error: {abs(mi_estimate_image_7/mi_estimate_image_28 - mutual_information_7):.2f}")