import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import bmi
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.estimators.neural._critic import (
    ConcatCritic, SeparableCritic, ConvolutionalCritic
)


class Discriminator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class CombinedArchitecture(nn.Module):
    def __init__(self, single_model, divergence):
        super(CombinedArchitecture, self).__init__()
        self.single_model = single_model
        self.divergence = divergence

    def forward(self, x, y):
        batch_size = x.size(0)
        xy = torch.cat([x, y], dim=1)
        x_y = torch.cat([x, y[torch.randperm(batch_size)]], dim=1)
        
        D_value_1 = self.single_model(xy)
        D_value_2 = self.single_model(x_y)
        
        return D_value_1, D_value_2

class DIMEEstimator:
    def __init__(
        self,
        batch_size=256,
        max_n_steps=2000,
        learning_rate=1e-3,
        hidden_layers=(256, 256),
        divergence='GAN',
        architecture='separable',
        alpha=1,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        x_shape=None,
        y_shape=None
    ):
        self.batch_size = batch_size
        self.max_n_steps = max_n_steps
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers
        self.divergence = divergence
        self.architecture = architecture
        self.alpha = alpha
        self.device = device
        self.critic = None
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.latent_dim = None

    def _create_critic(self, input_dim, output_dim=1):
        self.latent_dim = input_dim//2
        
        if self.architecture == 'joint':
            return ConcatCritic(self.latent_dim, 256, 2, 'relu', self.divergence).to(self.device)
        elif self.architecture == 'separable':
            return SeparableCritic(self.x_shape,self.y_shape, 256, 32, 2, 'relu', self.divergence).to(self.device)
        elif self.architecture == 'deranged':
            single_model = Discriminator(2 * self.latent_dim, 1)
            return CombinedArchitecture(single_model, self.divergence).to(self.device)
        elif self.architecture == 'conv_critic':
            return ConvolutionalCritic(self.divergence, None).to(self.device)

    def _compute_loss_ratio(self, D_value_1=None, D_value_2=None, scores=None):
        if self.divergence == 'KL':
            if self.architecture == 'deranged':
                loss, R, _ = self._kl_dime_deranged(D_value_1, D_value_2)
            else:
                loss, R = self._kl_dime_e(scores)
        elif self.divergence == 'GAN':
            if self.architecture == 'deranged':
                loss, R = self._gan_dime_deranged(D_value_1, D_value_2)
            else:
                loss, R = self._gan_dime_e(scores)
        elif self.divergence == 'HD':
            if self.architecture == 'deranged':
                loss, R = self._hd_dime_deranged(D_value_1, D_value_2)
            else:
                loss, R = self._hd_dime_e(scores)
        return loss, R

    def _kl_dime_deranged(self, D_value_1, D_value_2):
        eps = 1e-5
        batch_size_1 = D_value_1.size(0)
        batch_size_2 = D_value_2.size(0)
        valid_1 = torch.ones((batch_size_1, 1), device=self.device)
        valid_2 = torch.ones((batch_size_2, 1), device=self.device)
        loss_1 = self._my_binary_crossentropy(valid_1, D_value_1) * self.alpha
        loss_2 = self._wasserstein_loss(valid_2, D_value_2)
        loss = loss_1 + loss_2
        J_e = self.alpha * torch.mean(torch.log(D_value_1 + eps)) - torch.mean(D_value_2)
        VLB_e = J_e / self.alpha + 1 - np.log(self.alpha)
        R = D_value_1 / self.alpha
        return loss, R, VLB_e

    def _kl_dime_e(self, scores):
        eps = 1e-7
        scores_diag = scores.diag()
        n = scores.size(0)
        scores_no_diag = scores - scores_diag * torch.eye(n, device=self.device)
        loss_1 = -torch.mean(torch.log(scores_diag + eps))
        loss_2 = torch.sum(scores_no_diag) / (n*(n-1))
        loss = loss_1 + loss_2
        return loss, scores_diag

    def _gan_dime_deranged(self, D_value_1, D_value_2):
        BCE = nn.BCELoss()
        batch_size_1 = D_value_1.size(0)
        batch_size_2 = D_value_2.size(0)
        valid_2 = torch.ones((batch_size_2, 1), device=self.device)
        fake_1 = torch.zeros((batch_size_1, 1), device=self.device)
        loss_1 = BCE(D_value_1, fake_1)
        loss_2 = BCE(D_value_2, valid_2)
        loss = loss_1 + loss_2
        R = (1 - D_value_1) / D_value_1
        return loss, R

    def _gan_dime_e(self, scores):
        eps = 1e-5
        batch_size = scores.size(0)
        scores_diag = scores.diag()
        scores_no_diag = scores - scores_diag*torch.eye(batch_size, device=self.device) + torch.eye(batch_size, device=self.device)
        R = (1 - scores_diag) / scores_diag
        loss_1 = torch.mean(torch.log(torch.ones(scores_diag.shape, device=self.device) - scores_diag + eps))
        loss_2 = torch.sum(torch.log(scores_no_diag + eps)) / (batch_size*(batch_size-1))
        return -(loss_1+loss_2), R

    def _hd_dime_deranged(self, D_value_1, D_value_2):
        batch_size_1 = D_value_1.size(0)
        batch_size_2 = D_value_2.size(0)
        valid_1 = torch.ones((batch_size_1, 1), device=self.device)
        valid_2 = torch.ones((batch_size_2, 1), device=self.device)
        loss_1 = self._wasserstein_loss(valid_1, D_value_1)
        loss_2 = self._reciprocal_loss(valid_2, D_value_2)
        loss = loss_1 + loss_2
        R = 1 / (D_value_1 ** 2)
        return loss, R

    def _hd_dime_e(self, scores):
        eps = 1e-5
        Eps = 1e7
        scores_diag = scores.diag()
        n = scores.size(0)
        scores_no_diag = scores + Eps * torch.eye(n, device=self.device)
        loss_1 = torch.mean(scores_diag)
        loss_2 = torch.sum(torch.pow(scores_no_diag, -1))/(n*(n-1))
        loss = -(2 - loss_1 - loss_2)
        return loss, 1 / (scores_diag**2)

    def _my_binary_crossentropy(self, y_true, y_pred):
        eps = 1e-7
        return -torch.mean(torch.log(y_true)+torch.log(y_pred + eps))

    def _wasserstein_loss(self, y_true, y_pred):
        return torch.mean(y_true * y_pred)

    def _reciprocal_loss(self, y_true, y_pred):
        return torch.mean(1 / y_pred)

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
        pbar = tqdm(total=self.max_n_steps, desc="Training DIME")
        while steps < self.max_n_steps:
            for x_batch, y_batch in dataloader:
                if steps >= self.max_n_steps:
                    break
                
                if self.architecture == 'deranged':
                    D_value_1, D_value_2 = self.critic(x_batch, y_batch)
                    loss, R = self._compute_loss_ratio(D_value_1=D_value_1, D_value_2=D_value_2)
                else:
                    scores = self.critic(x_batch, y_batch)
                    loss, R = self._compute_loss_ratio(scores=scores)

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
        self.critic.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            Y = torch.tensor(Y, dtype=torch.float32).to(self.device)
            
            if self.architecture == 'deranged':
                D_value_1, D_value_2 = self.critic(X, Y)
                _, R = self._compute_loss_ratio(D_value_1, D_value_2)
            else:
                scores = self.critic(X, Y)
                _, R = self._compute_loss_ratio(scores=scores)
            mi_estimate = torch.log(R).mean().item()
        
        return mi_estimate

if __name__ == '__main__':
    results = []
    for task_name in bmi.benchmark.BENCHMARK_TASKS.keys():
        task = bmi.benchmark.BENCHMARK_TASKS[task_name]
        print(f"Task: {task.name}")
        print(f"Task {task.name} with dimensions {task.dim_x} and {task.dim_y}")
        print(f"Ground truth mutual information: {task.mutual_information:.2f}")
        X, Y = task.sample(100000, seed=42)
        X = X.__array__()
        Y = Y.__array__()
        test_x, test_y = task.sample(10000, seed=42)
        test_x = test_x.__array__()
        test_y = test_y.__array__()

        configs = [
            {'divergence': 'KL', 'architecture': 'separable'},
            {'divergence': 'KL', 'architecture': 'deranged'},
            {'divergence': 'GAN', 'architecture': 'separable'},
            {'divergence': 'GAN', 'architecture': 'deranged'},
            {'divergence': 'HD', 'architecture': 'separable'},
            {'divergence': 'HD', 'architecture': 'deranged'},
        ]
        
        max_n_steps = 10000

        for config in configs:
            print(f"\nTesting configuration: {config}")
            dime = DIMEEstimator(
                max_n_steps=max_n_steps,
                x_shape=task.dim_x,
                y_shape=task.dim_y,
                learning_rate=1e-3,
            )
            dime.fit(X, Y)
            mi_estimate = dime.estimate(test_x, test_y)
            print(f"DIME estimate: {mi_estimate:.4f}")
            print(f"Absolute error: {abs(mi_estimate - task.mutual_information):.4f}")
            results.append({
                'estimator': 'DIME',
                'task': task.name,
                'true_mi': task.mutual_information,
                "max_n_steps": max_n_steps,
                "batch_size": 256,
                "learning_rate": 1e-3,
                "hidden_layers": (256, 256),
                'mean_estimate': mi_estimate,
                'absolute_error': abs(mi_estimate - task.mutual_information),
                'divergence': config['divergence'],
                'architecture': config['architecture']
            })
    import json
    with open('dime_results.json', 'w') as f:
        json.dump(results, f, indent=2)
            