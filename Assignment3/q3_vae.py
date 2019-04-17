import torch
import torchvision
from torch import nn, optim

from score_fid import get_test_loader

class Encoder(nn.Module):
    def __init__(self, latent_size=100):
        super(Encoder, self).__init__()

        self.latent_size = latent_size

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ELU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(32, 64, 3),
            nn.ELU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(64, 256, 6),
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
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, padding=2),
            nn.ELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, 3, padding=4),
            nn.ELU(),
            nn.Conv2d(16, 3, 3, padding=2)
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
    elbo = -torch.nn.functional.mse_loss(output, target, reduction='sum')
    elbo += 0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
    return elbo / output.size(0)

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")


    vae = VAE()
    vae = vae.to(device)

    optimizer = optim.Adam(vae.parameters(), lr=3e-4)

    running_loss = 0

    svhn_loader = get_test_loader(64)
    for epoch in range(20):

        print(f"------- EPOCH {epoch} --------")

        for i, (x, _) in enumerate(svhn_loader):
            vae.train()
            optimizer.zero_grad()

            x = x.to(device)
            y, mu, logvar = vae(x)

            loss = -ELBO(y, x, mu, logvar)
            running_loss += loss

            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f"Training example {i + 1} / {len(svhn_loader)}. Loss: {running_loss}")
                running_loss = 0

    torch.save(vae.state_dict(), 'q3_vae_save.pth')

    # Generate new images
    z = torch.randn(64, 100, device=device)
    generated = vae.decoder(z)
    torchvision.utils.save_image(generated, 'vae-generated.png', normalize=True)
        
