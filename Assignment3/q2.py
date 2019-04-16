import torch
from torch import nn, optim
import numpy as np

from mnist_loader import get_data_loader

class Encoder(nn.Module):
    def __init__(self, latent_size=100):
        super(Encoder, self).__init__()

        self.latent_size = latent_size

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ELU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(32, 64, 3),
            nn.ELU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(64, 256, 5),
            nn.ELU()
        )

        self.final = nn.Linear(256, self.latent_size * 2)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.final(x)
        return x[..., self.latent_size:], x[..., :self.latent_size] # Return mu and logvar

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.start = nn.Linear(100, 256)

        self.conv = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(256, 64, 5, padding=4),
            nn.ELU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 32, 3, padding=2),
            nn.ELU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(32, 16, 3, padding=2),
            nn.ELU(),
            nn.Conv2d(16, 1, 3, padding=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.start(x)
        x = x.view(x.size(0), -1, 1, 1)
        return self.conv(x)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(logvar / 2)
        z = mu + std * torch.randn_like(std)
        return self.decoder(z), mu, logvar

def ELBO(output, target, mu, logvar):
    elbo = -torch.nn.functional.binary_cross_entropy(output, target, reduction='sum')
    elbo += 0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
    return elbo / output.size(0)

def validate(model, valid, device, method='elbo'):
    model.eval()
    nb_minibatch = 0
    metric = 0
    for i, x in enumerate(valid):
        x = x.to(device)
        y, mu, logvar = model(x)

        if method == 'elbo':
            metric += ELBO(y, x, mu, logvar)
        else:
            std = torch.exp(logvar / 2)[:, None, :].repeat(1, 200, 1)
            z_samples = mu[:, None, :] + std * torch.randn_like(std)
            metric_val = torch.mean(importance_sample_vae(model, x.view(x.size(0), -1), z_samples, device))
            print(f"{i} / {len(valid)} : {metric_val}")
            metric += metric_val

        nb_minibatch += 1
    return metric / nb_minibatch

def part1(device):
    train, valid, test = get_data_loader("binarized_mnist", 64)

    vae = VAE()
    vae = vae.to(device)

    optimizer = optim.Adam(vae.parameters(), lr=3e-4)

    for epoch in range(20):

        print(f"------- EPOCH {epoch} --------")

        for i, x in enumerate(train):
            vae.train()
            optimizer.zero_grad()

            x = x.to(device)
            y, mu, logvar = vae(x)

            loss = -ELBO(y, x, mu, logvar)

            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                with torch.no_grad():
                    valid_elbo = validate(vae, valid, device)
                print(f"Training example {i + 1} / {len(train)}. Validation ELBO: {valid_elbo}")

    torch.save(vae.state_dict(), 'vae_save.pth')

def importance_sample_vae(vae, x, z, device):
    """
    x: [M, D]
    z: [M, K, L]
    M: batch size (64)
    K: number of importance  (200)
    D: Input dimension (784)
    L: Latent code dimension (100)
    """
    assert x.dim() == 2 and z.dim() == 3
    M = x.size(0)
    K = z.size(1)
    D = x.size(1)
    L = z.size(2)
    
    z_flattened = z.view(-1, L)
    p_x_given_z = torch.mean(vae.decoder(z_flattened).view(M, K, -1), dim=-1).unsqueeze(-1) # (64, 200, 784) -> (64, 200, 1) (M, K, 1)

    q_z_given_x, _ = vae.encoder(x.view(M, 1, int(np.sqrt(D)), -1)) # Just take the mean?
    q_z_given_x = q_z_given_x[:, None, :] # (64, 1, 100) (M, 1, L)

    normal = torch.distributions.normal.Normal(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))
    p_z = normal.log_prob(z) # Compare to a unit gaussian or to something in the NN? (64, 200, 100) (M, K, L)

    print(f"p_x_given_z --> max: {p_x_given_z.max()}, min: {p_x_given_z.min()}, mean: {p_x_given_z.mean()}")
    print(f"p_z --> max: {p_z.max()}, min: {p_z.min()}, mean: {p_z.mean()}")
    print(f"q_z_given_x --> max: {q_z_given_x.max()}, min: {q_z_given_x.min()}, mean: {q_z_given_x.mean()}")
    return torch.log(1 / K * torch.sum(torch.mean(p_x_given_z * p_z / q_z_given_x, dim=2), dim=1))


def part2(device):
    train, valid, test = get_data_loader("binarized_mnist", 64)

    vae = VAE()
    vae.load_state_dict(torch.load('vae_save.pth', map_location=device))
    vae.eval()
    vae = vae.to(device)

    with torch.no_grad():
        metric = validate(vae, valid, device, method='importance')
        print(metric)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    part2(device)