import torch
import numpy as np


def generate_real_points(distribution_type: str = 'cluster',
                         num_points: int = 512) -> torch.Tensor:
    if distribution_type == 'cluster':
        x, y = generate_cluster(num_points)
    elif distribution_type == 'linear':
        x, y = generate_linear(num_points)
    elif distribution_type == 'sin':
        x, y = generate_sin(num_points)

    points = np.stack([x, y], axis=1)
    return torch.from_numpy(points)


def generate_cluster(num_points) -> (np.ndarray, np.ndarray):
    x = np.array([0.2]*(num_points//2) + [0.8]*(num_points//2), np.float32)
    y = np.array([0.2]*(num_points//4) +
                 [0.8]*(num_points//4) +
                 [0.2]*(num_points//4) +
                 [0.8]*(num_points//4), np.float32)

    x += np.random.normal(0, 0.05, num_points)
    y += np.random.normal(0, 0.05, num_points)
    return x, y


def generate_linear(num_points) -> (np.ndarray, np.ndarray):
    x = np.linspace(0.1, 0.9, num_points, dtype=np.float32)
    y = 0.9*x
    x += np.random.normal(0, 0.03, num_points)
    y += np.random.normal(0, 0.03, num_points)
    return x, y


def generate_sin(num_points) -> (np.ndarray, np.ndarray):
    x = np.linspace(0.2, 0.8, num_points, dtype=np.float32)
    y = (np.sin(x*6*np.pi) + 1) /2
    x += np.random.normal(0, 0.01, num_points)
    return x, y

