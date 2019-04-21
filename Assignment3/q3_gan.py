
import torch
import torchvision
from torch import nn
from typing import Tuple
from score_fid import get_test_loader
from torch.optim import Adam
import numpy as np
from q3_vae import Encoder, Decoder
class Discriminator(nn.Module):
    def __init__(self, latent_size=100):
        super(Discriminator, self).__init__()

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

        self.final = nn.Linear(256, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.conv(image)
        x = x.view(x.size(0), -1)
        x = self.final(x)
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

    def lossDiscriminator(self, reals: torch.Tensor, latents: torch.Tensor = None) -> torch.FloatTensor:
        """
        WGAN-GP Loss
        """
        if latents is None:
            latents = torch.randn([reals.shape[0], self.latent_size], device=device)
        fakes = self.generator(latents)
        fakes_score = self.discriminator(fakes)
        reals_score = self.discriminator(reals)
        loss = fakes_score.mean() - reals_score.mean()
        
        grad_penalty = self.gradient_pernalty(reals, fakes)
        loss += self.lambda_coefficient * grad_penalty
        return loss

    def lossGenerator(self, batch_size, device, latents: torch.Tensor = None) -> torch.FloatTensor:
        latents = torch.randn([batch_size, self.latent_size], device=device)
        fakes = self.generator(latents)
        fakes_score = self.discriminator(fakes)
        loss = -fakes_score.mean()
        return loss
        
    def gradient_pernalty(self, reals, fakes):
        """
        Inspired from "https://discuss.pytorch.org/t/gradient-penalty-with-respect-to-the-network-parameters/11944/2"
        """
        z = random_interpolation(reals, fakes)
        z.requires_grad_(True)

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
    

def random_interpolation(x, y):
    a = torch.rand((x.size(0), 1, 1, 1), device=x.device)
    # a = a.expand(x.size(0), x.size(1), x.size(2), x.size(3))
    return a * x + (1-a) * y


def visual_samples(gan, dimensions, device, svhn_loader, step=0):
    # Generate new images
    z = torch.randn(64, dimensions, device=device)
    generated = gan.generator(z)
    torchvision.utils.save_image(generated, f"images/gan/3.1gan-generated-{step}.png", normalize=True)


def disentangled_representation(gan, dimensions, device, epsilon = 3):
    #Sample from prior p(z) which is a Std Normal
    z = torch.randn(dimensions, device=device)
    
    #Copy this tensor times its number of dimensions and make perturbations on each dimension
    #The first element is the original sample
    z = z.repeat(dimensions+1, 1)
    for i, sample in enumerate(z[1:]):
        sample[i] += epsilon

    generated = gan.generator(z)
    torchvision.utils.save_image(generated, 'images/gan/3_2positive_eps.png', normalize=True)

    #Do the same with the negative epsilon
    epsilon = -2*epsilon
    for i, sample in enumerate(z[1:]):
        sample[i] += epsilon

    #Make a batch of the pertubations and pass it through the generator
    generated = gan.generator(z)
    torchvision.utils.save_image(generated, 'images/gan/3_2negative_eps.png', normalize=True)

def interpolation(gan, dimensions, device):
    # Interpolate in the latent space between z_0 and z_1
    z_0 = torch.randn(1,dimensions, device=device)
    z_1 = torch.randn(1,dimensions, device=device)
    z_a = torch.zeros([11,dimensions], device=device)
    
    for i in range(11):
        a = i/10
        z_a[i] = torch.lerp(z_0, z_1, a)

    generated = gan.generator(z_a)
    torchvision.utils.save_image(generated, 'images/gan/3_3latent.png', normalize=True)
    
    # Interpolate in the data space between x_0 and x_1
    x_0 = gan.generator(z_0)
    x_1 = gan.generator(z_1)
    x_a = torch.zeros(11,x_0.size()[1],x_0.size()[2],x_0.size()[3], device=device)

    for i in range(11):
        a = i/10
        x_a[i] = torch.lerp(x_0, x_1, a)

    torchvision.utils.save_image(x_a, 'images/gan/3_3data.png', normalize=True)


def save_1000_images(img_dir: str):
    import os
    gan = GAN()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gan.load_state_dict(torch.load('q3_gan_save.pth', map_location=device))
    gan = gan.to(device)
    gan.eval()
    
    for p in gan.parameters():
        p.requires_grad = False

    for i in range(10):
        print(i)
        latents = torch.randn(100, 100, device=device)
        images = gan.generator(latents)
        os.makedirs(f"{img_dir}/img/", exist_ok=True)
        for j, image in enumerate(images):
            filename = f"{img_dir}/img/{i * 100 + j:03d}.png"
            torchvision.utils.save_image(image, filename, normalize=True)


if __name__ == '__main__':    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    gan = GAN()
    gan = gan.to(device)
    gan.train()

    disc_steps_per_gen_step = 5

    optimizerDiscrimator = Adam(gan.discriminator.parameters())
    optimizerGenerator = Adam(gan.generator.parameters())

    svhn_loader = get_test_loader(64)
    try: 
        gan.load_state_dict(torch.load('q3_gan_save.pth', map_location=device))
        print('----Using saved model----')

    except FileNotFoundError:
        for epoch in range(20):
            print(f"------- EPOCH {epoch} --------")

            running_loss_discriminator = 0
            running_loss_generator = 0
            
            for i, (real_images, _) in enumerate(svhn_loader):

                #Train the discriminator for a couple iterations
                optimizerDiscrimator.zero_grad()
                real_images = real_images.to(device)
                loss_disc = gan.lossDiscriminator(real_images)
                running_loss_discriminator += loss_disc
                loss_disc.backward()
                optimizerDiscrimator.step()


                #Then train the generator
                if i % disc_steps_per_gen_step == 0:
                    optimizerGenerator.zero_grad()
                    loss_gen = gan.lossGenerator(real_images.shape[0], device)
                    running_loss_generator += loss_gen
                    loss_gen.backward()
                    optimizerGenerator.step()

                if i % 100 == 0:
                    print(f"Training example {i} / {len(svhn_loader)}. DiscLoss: {running_loss_discriminator:.2f}, GenLoss: {running_loss_generator:.2f}")
                    running_loss_discriminator = 0
                    running_loss_generator = 0
            if epoch % 5 == 0:
                visual_samples(gan, 100, device, svhn_loader, step=epoch)
            
        torch.save(gan.state_dict(), 'q3_gan_save.pth')

    dimensions = 100
    
    gan.eval()
    #3.1 Visual samples
    visual_samples(gan, dimensions, device, svhn_loader)

    #3.2 Disentangled representation
    disentangled_representation(gan, dimensions, device, epsilon=10)

    #3.3 Interpolation
    interpolation(gan, dimensions, device)

    img_dir = "images/gan/fid"
    save_1000_images(img_dir)
