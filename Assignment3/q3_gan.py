
import torch
from torch import nn
from typing import Tuple

# TODO: change to q3_vae, rather than q2.
from q3_vae import Encoder, Decoder

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # It doesn't really matter what size we use, since we map down to a
        # scalar in the end.
        self.encoder = Encoder(100)
        self.final_layer = nn.Linear(100*2, 1)
        self.activation = nn.Sigmoid()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        mu, sigma = self.encoder(image)
        x = torch.cat((mu, sigma), -1)
        x = self.final_layer(x)
        x = self.activation(x)
        return x

class Generator(Decoder):
    pass


class GAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_size = 100
        self.generator = Generator()
        self.discriminator = Discriminator()

        self.lambda_coefficient = 10.0

    def loss(self, reals: torch.Tensor, latents: torch.Tensor = None) -> torch.FloatTensor:
        """
        WGAN-GP Loss
        """
        if latents is None:
            latents = torch.randn([reals.shape[0], self.latent_size])
        fakes = self.generator(latents)
        fakes_score = self.discriminator(fakes)
        reals_score = self.discriminator(reals)
        loss = (fakes_score + reals_score).mean()
        
        grad_penalty = self.gradient_pernalty(reals, fakes)
        loss += self.lambda_coefficient * grad_penalty
        return loss

        
    def gradient_pernalty(self, reals, fakes):
        """
        Inspired from "https://discuss.pytorch.org/t/gradient-penalty-with-respect-to-the-network-parameters/11944/2"
        """
        z = random_interpolation(reals, fakes)
        z.requires_grad = True

        output = disc(z)
        gradients = torch.autograd.grad(
            outputs=output,
            inputs=z,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        gradient = gradients[0]
        norm_2 = gradient.norm(p=2)
        return ((norm_2 - 1)**2).mean()

def random_interpolation(x, y):
    a = torch.rand_like(x)
    return a * x + (1-a) * y

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    from score_fid import get_test_loader
    from torch.optim import Adam
    
    gan = GAN()

    optimizer = Adam(gan.parameters(), lr=3e-4)

    running_loss = 0
    for epoch in range(20):
        svhn_loader = get_test_loader(64)
        print(f"------- EPOCH {epoch} --------")

        for i, (real_images, _) in enumerate(svhn_loader):
            gan.train()
            optimizer.zero_grad()

            real_images = real_images.to(device)
            loss = gan.loss(real_images)
            running_loss += loss

            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f"Training example {i + 1} / {len(svhn_loader)}. Loss: {running_loss}", end="\r\n")
                running_loss = 0