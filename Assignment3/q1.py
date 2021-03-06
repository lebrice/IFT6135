#!~/Source/IFT6135/env/bin/python3
try:
        
    import torch
    from torch import nn
    import numpy as np
    import progressbar
    from typing import Iterable
    import matplotlib.pyplot as plt
except ModuleNotFoundError as e:
    print(e)
    print("Please install the modules listed in requirements-pip.txt with: 'pip install -r requirements-pip.txt'")


cuda = torch.cuda.is_available()
print("cuda?", cuda)

def jensen_shannon_divergence(network: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.FloatTensor:
    D_theta_x = torch.sigmoid(network(x))
    D_theta_y = torch.sigmoid(network(y))
    log_2 = torch.as_tensor(np.log(2)).to(x.device)
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
    # gradient = gradients.view(gradients.size(0), -1)
    norm_2 = gradient.norm(p=2, dim=1)
    return ((norm_2 - 1)**2).mean()


def maximize_objective(objective, p, q, network=None, maxsteps=1000, threshold=0.001):
    p = to_tensors(p)
    q = to_tensors(q)

    p_0 = next(p)
    q_0 = next(q)
    assert p_0.shape == q_0.shape, "P and Q have to have the same shape of elements!"

    network = network if network is not None else nn.Sequential(
        nn.Linear(p_0.shape[-1], 512),
        nn.Tanh(),
        nn.Linear(512, 256),
        nn.Tanh(),
        nn.Linear(256, 1),
    )

    if cuda:
        network = network.cuda()
    
    network.train()
    network.float()
    params = network.parameters()
    optimizer = torch.optim.Adam(params, lr=1e-3)

    value: float = 0.0
    hook = StopIfConverged(threshold=threshold)
    with progressbar.ProgressBar(max_value=maxsteps, prefix=f"{objective.__name__}", redirect_stdout=True) as bar:
        for i, x, y in zip(range(maxsteps), p, q):
            if cuda:
                x = x.cuda()
                y = y.cuda()

            loss = - objective(network, x, y)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            value = - loss.item()
            if hook(value):
                # print(f"converged at step {i}")
                break
            
            bar.update(i)
        else:
            print(f"Did not converge after {i} steps!")
            pass
    # print(f"Steps: {i}, value: {value}")
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
        t = torch.as_tensor(item).float()
        if cuda:
            t = t.cuda()
        yield t


def get_optimal_discriminator(f_0, f_1, **kwargs):
    def discriminator_value_fn(network: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        D_theta_x = torch.log(network(x))
        D_theta_y = torch.log(1 - network(y))
        return (D_theta_x + D_theta_y).sum()

    discriminator_network = nn.Sequential(
        nn.Linear(1, 512),
        nn.Tanh(),
        nn.Linear(512, 256),
        nn.Tanh(),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )

    value, disc = maximize_objective(
        discriminator_value_fn,
        network=discriminator_network,
        p=f_0,
        q=f_1,
        **kwargs,
    )

    def disc_numpy(x: np.ndarray) -> np.ndarray:
        """
        Adds preprocessing and postprocessing to/from numpy
        """
        with torch.no_grad():
            _x = torch.as_tensor(x).view([-1, 1]).float()
            if cuda:
                _x = _x.cuda()
            d_x = disc(_x)
            d_x = np.reshape(d_x.cpu().numpy(), x.shape)
            return d_x

    return disc_numpy


def q1(p, q, **kwargs):
    return maximize_objective(jensen_shannon_divergence, p, q, **kwargs)


def q2(p, q, **kwargs):
    return maximize_objective(wasserstein_distance, p, q, **kwargs)

def q3():
    print("Starting Q3.")
    from samplers import distribution1
    
    def get_samples(phi: float):
        p = distribution1(0)
        q = distribution1(phi)
        jsd, _ = q1(p, q, maxsteps=1000, threshold=0.01)
        wd, _  = q2(p, q, maxsteps=1000, threshold=0.01)
        return jsd, wd

    phis = np.arange(-1, 1.1, 0.1)
    jsds = []; wds = []
    for i, phi in enumerate(phis):
        jsd, wd = get_samples(phi)
        print(f"phi: {phi:.2f}, jsd: {jsd:.2f}, wd: {wd:.2f}")
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
    plt.savefig("./images/q1_3.png")
    plt.show()


def q4():
    import density_estimation
    plt.show()

if __name__ == "__main__":
    q3()
    q4()
    