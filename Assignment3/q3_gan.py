
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
        self.batch_size = None

        # TODO: Tune this hyperparameter
        self.d_steps_per_g_step = 3

        self.training_the_discriminator = True

    def g_loss(self) -> torch.FloatTensor:
        assert self.batch_size is not None, "d_loss should have been called before g_loss"
        latents = torch.randn([self.batch_size, self.latent_size])
        fakes = self.generator(latents)
        fakes_scores = self.discriminator(fakes)
        return - fakes_scores.mean()

    def d_loss(self, reals: torch.Tensor) -> torch.FloatTensor:
        """
        WGAN-GP Loss
        """
        self.batch_size = reals.shape[0]
        latents = torch.randn([self.batch_size, self.latent_size])
        fakes = self.generator(latents)
        fakes_score = self.discriminator(fakes)
        reals_score = self.discriminator(reals)
        loss = (fakes_score - reals_score).mean()
        
        grad_penalty = self.gradient_pernalty(reals, fakes)
        loss += self.lambda_coefficient * grad_penalty
        return loss

        
    def gradient_pernalty(self, reals, fakes):
        """
        Inspired from "https://discuss.pytorch.org/t/gradient-penalty-with-respect-to-the-network-parameters/11944/2"
        """
        z = random_interpolation(reals, fakes)

        output = self.discriminator(z)
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

    @property
    def training_the_discriminator(self) -> bool:
        return self._training_the_discriminator
    
    @training_the_discriminator.setter
    def training_the_discriminator(self, value: bool) -> None:


    def training_generator(self) -> None:
       

    def training_discriminator(self) -> None:
        if not self._training_discriminator:
            for p in self.generator.parameters():
                p.requires_grad = False
            for p in self.discriminator.parameters():
                p.requires_grad = True


def random_interpolation(x, y):
    a = torch.rand_like(x)
    return a * x + (1-a) * y


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    import torchvision
    from score_fid import get_test_loader
    from torch.optim import Adam
    
    gan = GAN()
    gan = gan.to(device)
    gan.train()

    g_optimizer = Adam(gan.generator.parameters(), lr=3e-4)
    d_optimizer = Adam(gan.discriminator.parameters(), lr=3e-4)
    
    running_d_loss = 0
    running_g_loss = 0

    svhn_loader = get_test_loader(64)
    num_batches = len(svhn_loader)

    for epoch in range(2):
        print(f"------- EPOCH {epoch} --------")
        
        i = 0

        image_gen = (images for images, _ in svhn_loader)
        for i, image_batch in enumerate(image_gen):
            gan.training_discriminator()
            for t in range(gan.d_steps_per_g_step):
                print(t, image_batch.shape)
                d_optimizer.zero_grad()
                d_loss = gan.d_loss(image_batch)
                d_loss.backward()
                d_optimizer.step()

                running_d_loss += d_loss

                if (i + 1) % 100 == 0:
                    print(f"Training example {i + 1} / {num_batches}. D-Loss: {running_d_loss:.2f}, G-Loss: {running_g_loss:.2f}")
                    running_loss = 0
                i += 1

            gan.training_generator()
            g_optimizer.zero_grad()
            g_loss = gan.g_loss()
            g_loss.backward()
            g_optimizer.step()
            running_g_loss += g_loss

    torch.save(gan.state_dict(), 'q3_gan_save.pth')

    # Generate new images
    z = torch.randn(64, 100, device=device)
    generated = gan.generator(z)
    torchvision.utils.save_image(generated, 'gan-generated.png', normalize=True)
        
