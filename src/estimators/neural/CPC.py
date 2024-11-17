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

class CPCEstimator:
    def __init__(
        self,
        batch_size=256,
        max_n_steps=2000,
        learning_rate=1e-4,
        hidden_layers=(100, 100),
        temperature=1.0,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        x_shape=None,
        y_shape=None
    ):
        self.batch_size = batch_size
        self.max_n_steps = max_n_steps
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers
        self.temperature = temperature
        self.device = device
        self.critic = None
        self.x_shape = x_shape
        self.y_shape = y_shape

    def _create_critic(self, input_dim):
        return MLP(input_dim, self.hidden_layers).to(self.device)

    def _create_image_critic(self, x_channels, y_channels):
        return ConvCritic(x_channels, y_channels).to(self.device)

    @staticmethod
    def _infonce_lower_bound(scores, temperature):
        positive_samples = torch.diag(scores)
        nll = -positive_samples + torch.logsumexp(scores / temperature, dim=1)
        mi = torch.mean(-nll) + torch.log(torch.tensor(scores.shape[0]))
        return mi

    def fit(self, X: np.ndarray, Y: np.ndarray):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        Y = torch.tensor(Y, dtype=torch.float32).to(self.device)

        if self.critic is None:
            input_dim = X.shape[1] + Y.shape[1]
            self.critic = self._create_critic(input_dim)

        optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        steps = 0

        pbar = tqdm(total=self.max_n_steps, desc="Training CPC")
        while steps < self.max_n_steps:
            for x_batch, y_batch in dataloader:
                if steps >= self.max_n_steps:
                    break

                scores = torch.zeros(self.batch_size, self.batch_size).to(self.device)
                for i in range(self.batch_size):
                    scores[i] = self.critic(x_batch[i].unsqueeze(0).repeat(self.batch_size, 1), y_batch).squeeze()

                mi_estimate = self._infonce_lower_bound(scores, self.temperature)
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
            n_samples = X.shape[0]
            scores = torch.zeros(n_samples, n_samples).to(self.device)
            for i in range(n_samples):
                scores[i] = self.critic(X[i].unsqueeze(0).repeat(n_samples, 1), Y).squeeze()
            final_mi = self._infonce_lower_bound(scores, self.temperature).item()

        print(f"Final estimate - MI: {final_mi}")
        return final_mi

    def fit_image(self, X: np.ndarray, Y: np.ndarray):
        if X.ndim == 3:
            X = X.unsqueeze(1) 
        if Y.ndim == 3:
            Y = Y.unsqueeze(1)

        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        Y = torch.tensor(Y, dtype=torch.float32).to(self.device)

        if self.critic is None:
            self.critic = self._create_image_critic(X.shape[1], Y.shape[1])

        optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        steps = 0

        pbar = tqdm(total=self.max_n_steps, desc="Training CPC for image data")
        while steps < self.max_n_steps:
            for x_batch, y_batch in dataloader:
                if steps >= self.max_n_steps:
                    break

                scores = torch.zeros(self.batch_size, self.batch_size).to(self.device)
                for i in range(self.batch_size):
                    x_repeated = x_batch[i].unsqueeze(0).repeat(self.batch_size, 1, 1, 1)
                    scores[i] = self.critic(x_repeated, y_batch).squeeze()

                mi_estimate = self._infonce_lower_bound(scores, self.temperature)
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
            n_samples = X.shape[0]
            scores = torch.zeros(n_samples, n_samples).to(self.device)
            for i in range(n_samples):
                x_repeated = X[i].unsqueeze(0).repeat(n_samples, 1, 1, 1)
                scores[i] = self.critic(x_repeated, Y).squeeze()
            final_mi = self._infonce_lower_bound(scores, self.temperature).item()

        print(f"Final estimate for image data - MI: {final_mi}")
        return final_mi

if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../benchmark/self-consistency')))
    from self_consistency_benchmark import Self_Consistency_Benchmark

    cpc_image = CPCEstimator(max_n_steps=20,batch_size=16)
    
    baseline_task = Self_Consistency_Benchmark(task_type='baseline', dataset='mnist', rows=7)
    X, Y = baseline_task.sample(10000, seed=42)
    cpc_image.fit_image(X, Y)
    X, Y = baseline_task.sample(64, seed=42)
    mi_estimate_image = cpc_image.estimate_image(X, Y) 
    mutual_information = baseline_task.mutual_information

    baseline_task = Self_Consistency_Benchmark(task_type='baseline', dataset='mnist', rows=28)
    X, Y = baseline_task.sample(10000, seed=42)
    cpc_image.fit_image(X, Y)
    X, Y = baseline_task.sample(64, seed=42)
    mi_estimate_image_all = cpc_image.estimate_image(X, Y) 

    print(f"CPC estimate for image data: {mi_estimate_image/mi_estimate_image_all:.2f}")
    print(f"Ground truth mutual information: {mutual_information:.2f}")
    print(f"Absolute error: {abs(mi_estimate_image/mi_estimate_image_all - mutual_information):.2f}")