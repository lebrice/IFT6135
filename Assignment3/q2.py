import torch
from torch import nn, optim

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
    elbo = -torch.nn.functional.binary_cross_entropy(output, target)
    elbo += 0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
    return loelboss

def validate(model, valid, device):
    elbo = 0
    for x in valid:
        x = x.to(device)
        y, mu, logvar = model(x)
        elbo += ELBO(y, x, mu, logvar)
    return elbo

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    train, valid, test = get_data_loader("binarized_mnist", 64)

    vae = VAE()
    vae = vae.to(device)

    optimizer = optim.Adam(vae.parameters(), lr=3e-4)

    for epoch in range(20):

        print(f"------- EPOCH {epoch} --------")

        for i, x in enumerate(train):
            optimizer.zero_grad()

            x = x.to(device)
            y, mu, logvar = vae(x)

            loss = -ELBO(y, x, mu, logvar)

            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                with torch.no_grad():
                    total_elbo = validate(vae, valid, device)
                print(f"Training example {i + 1} / {len(train)}. Validation ELBO: {total_elbo}")
