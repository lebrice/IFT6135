#!~/Source/IFT6135/env/bin/python3

import torch
from torch import nn
from torch.optim import SGD
import numpy as np
from typing import Iterable


def jensen_shannon_divergence(network: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.FloatTensor:
    D_theta_x = torch.sigmoid(network(x))
    D_theta_y = torch.sigmoid(network(y))
    log_2 = torch.as_tensor(np.log(2))
    return log_2 + \
        0.5 * torch.log(D_theta_x).mean() + \
        0.5 * torch.log(1- D_theta_y).mean()


def wasserstein_distance(network: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.FloatTensor:
    T_theta_x = network(x)
    T_theta_y = network(y)
    obj = T_theta_x.mean()
    obj -= T_theta_y.mean()

    lambda_coefficient = 10.0
    obj -= lambda_coefficient * gradient_pernalty(network, x, y)
    return obj


def gradient_pernalty(model, x, y):
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
    return ((norm_2 - 1)**2).mean()


def maximize_objective(objective, p, q, maxsteps=5000, threshold=0.001):
    p = to_tensors(p)
    q = to_tensors(q)

    p_0 = next(p)
    q_0 = next(q)
    assert p_0.shape == q_0.shape, "P and Q have to have the same shape of elements!"

    network = nn.Sequential(
        nn.Linear(p_0.shape[-1], 512),
        nn.Linear(512, 256),
        nn.Linear(256, 1),
    )

    cuda = torch.cuda.is_available()
    if cuda:
        network = network.cuda()
    
    network.train()
    network.float()
    params = network.parameters()
    optimizer = SGD(params, lr=1e-3)

    value: float = 0.0
    hook = StopIfConverged(threshold=threshold)
    for i, x, y in zip(range(maxsteps), p, q):
        # print(x)
        loss = - objective(network, x, y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        value = - loss.item()
        if hook(value):
            print(f"converged at step {i}")
            break
    else:
        print(f"Did not converge after {i} steps!")

    print(f"Steps: {i}, value: {value}")
    return value, network


class NumpyBuffer(object):
    """FIFO buffer with numpy."""
    def __init__(self, capacity: int):
        self.x = np.zeros([capacity])
    
    def put(self, new_value):
        self.x[:-1] = self.x[1:]; self.x[-1] = new_value


class StopIfConverged(object):
    """
    Little helper class that helps detect when we have converged and save us some time.

    Parameters
    ----------
    patience :  int, optional. The number of values to keep in the buffer.
    threshold : float, optional. The threshold on the total variation and the std.
    maximizing: bool, optional. Whether we are maximizing or minimizing the objective.

    Returns
    ---------
    converged: bool, Wether or not we have converged, as detected using the threshold and patience.
    """
    def __init__(self, patience=50, threshold=0.001, maximizing=True):
        self.patience = patience
        self.threshold = threshold
        self.maximizing = maximizing
        self._values = NumpyBuffer(patience)
        self._steps = 0

    def __call__(self, value):
        self._values.put(value)
        
        if self._steps < self.patience:
            self._steps += 1
            # we haven't filled the buffer, hence we have not converged.
            return False

        total_diff = np.sum(np.diff(self._values.x))
        if self.maximizing:
            # if the objective is still increasing, then we have not converged.
            if total_diff >= self.threshold:
                return False
        else:
            # if the objective is still decreasing, then we have not converged.
            if total_diff <= self.threshold:
                return False

        # If we have stagnated, then we have converged.
        std = np.std(self._values.x)
        return std <= self.threshold
            
        


def gaussian_distribution(mean=0, std=1, mini_batch_size=512) -> np.ndarray:
    """
    A test distribution.
    """
    while True:
        x = np.random.normal(mean, std, [mini_batch_size, 1])
        x = torch.as_tensor(x).float()
        yield x


def to_tensors(generator: Iterable[np.ndarray]) -> Iterable[torch.Tensor]:
    for item in generator:
        yield torch.as_tensor(item).float()


def q1(p, q):
    return maximize_objective(jensen_shannon_divergence, p, q)


def q2(p, q):
    return maximize_objective(wasserstein_distance, p, q)


def q3():
    from samplers import distribution1
    import matplotlib.pyplot as plt
    
    def get_samples(phi: float):
        p = distribution1(0)
        q = distribution1(phi)
        jsd, _ = q1(p, q)
        wd, _ = q2(p, q)
        return jsd, wd

    phis = np.arange(-1, 1.1, 0.1)
    jsds = []; wds = []
    for i, phi in enumerate(phis):
        print(f"phi: {phi:.2f}")
        jsd, wd = get_samples(phi)
        jsds.append(jsd)
        wds.append(wd)

    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot(1, 1, 1)

    ax.scatter(phis, jsds, label="JSD")
    ax.scatter(phis, wds, label="WD")
    ax.set_title("WD and JSD vs phi")
    ax.legend()
    ax.set_xlabel("phi")
    ax.set_ylabel("Distance metric value")
    plt.show()
    plt.savefig("./q1_3.png")


def get_optimal_discriminator(f_0, f_1):
    def discriminator_value_fn(network: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        D_theta_x = nn.functional.logsigmoid(network(x))
        D_theta_y = torch.log(1 - torch.sigmoid(network(y)))
        return (D_theta_x + D_theta_y).mean()

    value, discriminator = maximize_objective(
        discriminator_value_fn,
        p=f_0,
        q=f_1,
        maxsteps=10000,
        threshold=0.001,
    )
    def disc(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = torch.as_tensor(x).view([-1, 1]).float()
            d_x = torch.sigmoid(discriminator(x))
            return d_x.numpy()

    return disc

def q4():
    import matplotlib.pyplot as plt
    import density_estimation
    plt.show()
    plt.savefig("./q1_4.png")

if __name__ == "__main__":
    q4()
    