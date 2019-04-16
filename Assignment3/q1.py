#!~/Source/IFT6135/env/bin/python3

import torch
from torch import nn
from torch.optim import SGD
import numpy as np

def jensen_shannon_divergence(p, q):
    """
    TODO: Really unsure about this. Are we supposed to train a network that gives the JSD for two particular distributions?
    or for any two distributions?
    """
    D_theta = nn.Sequential(
        nn.Linear(1, 512),
        nn.Linear(512, 256),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )

    cuda = torch.cuda.is_available()
    print("cuda?", cuda)
    if cuda:
        D_theta = D_theta.cuda()
    
    D_theta.train()
    D_theta.float()
    params = D_theta.parameters()
    optimizer = SGD(params, lr=1e-3)

    def objective(D_theta: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.FloatTensor:
        D_theta_x = D_theta(x)
        D_theta_y = D_theta(y)
        log_2 = torch.as_tensor(np.log(2))
        return log_2 + \
            0.5 * torch.log(D_theta_x).mean() + \
            0.5 * torch.log(1- D_theta_y).mean()

    for i, x, y in zip(range(1000), p, q):
        # print(x)
        loss = - objective(D_theta, x, y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        print(i, loss.item())
        # mean_D_theta_x = D_theta_x.
        # print(i, D_theta_x, D_theta_y)

    D_theta.eval()
    return D_theta


def wasserstein_distance(p, q):
    """
    Also really unsure about this.
    """
    T_theta = nn.Sequential(
        nn.Linear(1, 512),
        nn.Linear(512, 256),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )

    cuda = torch.cuda.is_available()
    print("cuda?", cuda)
    if cuda:
        T_theta = T_theta.cuda()
    
    T_theta.train()
    T_theta.float()
    params = T_theta.parameters()
    optimizer = SGD(params, lr=1e-3)
    
    

    def gradient_pernalty(model, x, y, lambda_coefficient):
        """
        Inspired from "https://discuss.pytorch.org/t/gradient-penalty-with-respect-to-the-network-parameters/11944/2"
        """
        def random_interpolation(x, y):
            a = np.random.uniform(0, 1)
            return a * x + (1-a) * y

        z = random_interpolation(x, y)
        z.requires_grad = True

        output = model(z)
        gradients = torch.autograd.grad(
            outputs=output,
            inputs=z,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        gradient = gradients[0]
        norm_2 = gradient.norm(dim=1, p=2)
        return lambda_coefficient * ((norm_2 - 1)**2).mean()

    def objective(T_theta: nn.Module, x: torch.Tensor, y: torch.Tensor, lambda_coefficient) -> torch.FloatTensor:
        T_theta_x = T_theta(x)
        T_theta_y = T_theta(y)
        obj = T_theta_x.mean() - T_theta_y.mean()
        obj -= gradient_pernalty(T_theta, x, y, lambda_coefficient)
        return obj 
       
    for i, x, y in zip(range(1000), p, q):
        lambda_coefficient = 10.0
        loss = - objective(T_theta, x, y, lambda_coefficient)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        print(i, loss.item())
        # break
        # mean_D_theta_x = D_theta_x.
        # print(i, D_theta_x, D_theta_y)

    T_theta.eval()
    return T_theta

def gaussian_distribution(mean=0, std=1, mini_batch_size=512) -> np.ndarray:
    """
    A test distribution.
    """
    while True:
        x = np.random.normal(mean, std, [mini_batch_size, 1])
        x = torch.as_tensor(x).float()
        yield x

if __name__ == "__main__":
    # from samplers import distribution1, distribution2
    p = gaussian_distribution(0, 1)
    q = gaussian_distribution(5, 1)
    # div_function = jensen_shannon_divergence(p, q)
    div_function = wasserstein_distance(p, q)
    