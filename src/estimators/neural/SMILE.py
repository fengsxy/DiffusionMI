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

class SMILEEstimator:
    def __init__(
        self,
        batch_size=256,
        max_n_steps=2000,
        learning_rate=1e-4,
        hidden_layers=(100, 100),
        clip_value=20.0,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        x_shape=None,
        y_shape=None
    ):
        self.batch_size = batch_size
        self.max_n_steps = max_n_steps
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers
        self.clip_value = clip_value
        self.device = device
        self.critic = None
        self.x_shape = x_shape
        self.y_shape = y_shape

    def _create_critic(self, input_dim):
        return MLP(input_dim, self.hidden_layers).to(self.device)

    def _create_image_critic(self, x_channels, y_channels):
        return ConvCritic(x_channels, y_channels).to(self.device)

    @staticmethod
    def _smile_lower_bound(t, t_shuffle, clip_value):
        t_clip = t
        t_shuffle_clip = torch.clamp(t_shuffle, -clip_value, clip_value)
        js = torch.mean(-torch.nn.functional.softplus(-t_clip)) - \
            torch.mean(torch.nn.functional.softplus(t_shuffle_clip))
        dv = torch.mean(t) - torch.log(torch.mean(torch.exp(t_shuffle)))
        return js + (dv - js).detach()

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

        pbar = tqdm(total=self.max_n_steps, desc="Training SMILE")
        while steps < self.max_n_steps:
            for x_batch, y_batch in dataloader:
                if steps >= self.max_n_steps:
                    break

                y_shuffle = y_batch[torch.randperm(y_batch.size(0))]
                t = self.critic(x_batch, y_batch)
                t_shuffle = self.critic(x_batch, y_shuffle)
                mi_estimate = self._smile_lower_bound(t, t_shuffle, self.clip_value)

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
            mi_estimate = self._smile_lower_bound(t, t_shuffle, self.clip_value)

        print(f"Final estimate - MI: {mi_estimate.item()}")
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

        pbar = tqdm(total=self.max_n_steps, desc="Training SMILE for image data")
        while steps < self.max_n_steps:
            for x_batch, y_batch in dataloader:
                if steps >= self.max_n_steps:
                    break

                y_shuffle = y_batch[torch.randperm(y_batch.size(0))]
                t = self.critic(x_batch, y_batch)
                t_shuffle = self.critic(x_batch, y_shuffle)
                mi_estimate = self._smile_lower_bound(t, t_shuffle, self.clip_value)

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
            mi_estimate = self._smile_lower_bound(t, t_shuffle, self.clip_value)

        print(f"Final estimate for image data - MI: {mi_estimate.item()}")
        return mi_estimate.item()

if __name__ == '__main__':
    # Test on normal data
    task = bmi.benchmark.BENCHMARK_TASKS['1v1-normal-0.75']
    print(f"Task: {task.name}")
    print(f"Task {task.name} with dimensions {task.dim_x} and {task.dim_y}")
    print(f"Ground truth mutual information: {task.mutual_information:.2f}")

    X, Y = task.sample(10000, seed=42)
    X = X.__array__()
    Y = Y.__array__()

    smile = SMILEEstimator(max_n_steps=2000)
    smile.fit(X, Y)
    mi_estimate = smile.estimate(X, Y)

    print(f"SMILE estimate for normal data: {mi_estimate:.2f}")
    print(f"Absolute error: {abs(mi_estimate - task.mutual_information):.2f}")

    # Test on image data
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../benchmark/self-consistency')))
    from self_consistency_benchmark import Self_Consistency_Benchmark

    smile_image = SMILEEstimator(max_n_steps=20)
    
    baseline_task = Self_Consistency_Benchmark(task_type='baseline', dataset='mnist', rows=7)
    X, Y = baseline_task.sample(10000, seed=42)
    smile_image.fit_image(X, Y)
    X, Y = baseline_task.sample(64, seed=42)
    mi_estimate_image = smile_image.estimate_image(X, Y) 
    mutual_information = baseline_task.mutual_information

    baseline_task = Self_Consistency_Benchmark(task_type='baseline', dataset='mnist', rows=28)
    X, Y = baseline_task.sample(10000, seed=42)
    smile_image.fit_image(X, Y)
    X, Y = baseline_task.sample(64, seed=42)
    mi_estimate_image_all = smile_image.estimate_image(X, Y) 

    print(f"SMILE estimate for image data: {mi_estimate_image:.2f}")
    print(f"Ground truth mutual information: {mutual_information:.2f}")
    print(f"Absolute error: {abs(mi_estimate_image/mi_estimate_image_all - mutual_information):.2f}")