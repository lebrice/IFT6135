import torch
import torchvision
from torch import nn
from typing import Tuple
from torch.optim import Adam
import torch.autograd as autograd
from classify_svhn import get_data_loader, get_data_loaderNoNormalize

#The implementation for the size of the Generator and the Discriminator is inspired from the DcGAN:
#https://github.com/pytorch/examples/blob/master/dcgan/main.py
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.conv(image)
        x = x.view(-1, 1).squeeze(1)
        return x
        
#The same architecture is used for the VAE decoder
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.convTranspose = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ELU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ELU(),
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False)
        )

        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1, 1, 1)
        x = self.convTranspose(x)
        x = self.activation(x)
        return x


class GAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_size = 100
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.lambda_coefficient = 10.0
    
def gradient_penalty(reals, fakes, gan):
    """
    Inspired from "https://discuss.pytorch.org/t/gradient-penalty-with-respect-to-the-network-parameters/11944/2"
    """
    z = random_interpolation(reals, fakes)
    z.requires_grad_(True)

    output = gan.discriminator(z)
    gradients = torch.autograd.grad(
        outputs=output,
        inputs=z,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    gradients = gradients[0]
    gradient = gradients.view(gradients.size(0), -1)
    norm_2 = gradient.norm(p=2, dim=1)

    return ((norm_2 - 1)**2).mean()

def random_interpolation(x, y):
    a = torch.rand((x.size(0), 1, 1, 1), device=x.device)
    return a * x + (1-a) * y

def visual_samples(gan, dimensions, device, svhn_loader, step=0):
    # Generate new images
    z = torch.randn(64, dimensions, device=device)
    generated = gan.generator(z)
    torchvision.utils.save_image(generated, 'images/gan/3_1gan-generated.png', nrow=10, normalize=True)
    
def disentangled_representation(gan, dimensions, device, epsilon = 3):
    #Sample from prior p(z) which is a Std Normal
    z = torch.randn(dimensions, device=device)
    
    #Copy this tensor times its number of dimensions and make perturbations on each dimension
    #The first element is the original sample
    z = z.repeat(dimensions+1, 1)
    for i, sample in enumerate(z[1:]):
        sample[i] += epsilon

    generated = gan.generator(z)
    torchvision.utils.save_image(generated, 'images/gan/3_2positive_eps.png', nrow=10, normalize=True)

    #Do the same with the negative epsilon
    epsilon = -2*epsilon
    for i, sample in enumerate(z[1:]):
        sample[i] += epsilon

    #Make a batch of the pertubations and pass it through the generator
    generated = gan.generator(z)
    torchvision.utils.save_image(generated, 'images/gan/3_2negative_eps.png', nrow=10, normalize=True)

def interpolation(gan, dimensions, device):
    # Interpolate in the latent space between z_0 and z_1
    z_0 = torch.randn(1,dimensions, device=device)
    z_1 = torch.randn(1,dimensions, device=device)
    z_a = torch.zeros([11,dimensions], device=device)

    for i in range(11):
        a = i/10
        z_a[i] = a*z_0 + (1-a)*z_1

    generated = gan.generator(z_a)
    torchvision.utils.save_image(generated, 'images/gan/3_3latent.png', nrow=11, normalize=True)
    
    # Interpolate in the data space between x_0 and x_1
    x_0 = gan.generator(z_0)
    x_1 = gan.generator(z_1)
    x_a = torch.zeros(11,x_0.size()[1],x_0.size()[2],x_0.size()[3], device=device)

    for i in range(11):
        a = i/10
        x_a[i] = a*x_0 + (1-a)*x_1

    torchvision.utils.save_image(x_a, 'images/gan/3_3data.png', nrow=11, normalize=True)


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

    optimizerDiscrimator = Adam(gan.discriminator.parameters(), lr=3e-4, betas=(0.5, 0.9))
    optimizerGenerator = Adam(gan.generator.parameters(), lr=3e-4, betas=(0.5, 0.9))

    trainloader, validloader, testloader = get_data_loaderNoNormalize("svhn", 64)
    
    try: 
        gan.load_state_dict(torch.load('q3_gan_save.pth', map_location=device))
        print('----Using saved model----')

    except FileNotFoundError:
        for epoch in range(5):
            print(f"------- EPOCH {epoch} --------")

            running_loss_discriminator = 0
            running_loss_generator = 0
            
            for i, (real_images, _) in enumerate(trainloader):
                
                #Train the discriminator for a couple iterations
                optimizerDiscrimator.zero_grad()                
                real_images = real_images.to(device)
                latents = torch.randn([real_images.shape[0], gan.latent_size], device=device)
                fakes = gan.generator(latents).detach()
                
                fakes_score = gan.discriminator(fakes)
                fakes_score_mean = fakes_score.mean()
                fakes_score_mean.backward()

                reals_score = gan.discriminator(real_images)
                reals_score_mean = -reals_score.mean()
                reals_score_mean.backward()
                loss = fakes_score_mean + reals_score_mean
            
                grad_penalty = gan.lambda_coefficient * gradient_penalty(real_images, fakes, gan)
                grad_penalty.backward()
                loss += grad_penalty
                
                optimizerDiscrimator.step()
                running_loss_discriminator += loss

                #Then train the generator
                if i % disc_steps_per_gen_step == 0:
                    optimizerGenerator.zero_grad()
                    latents = torch.randn([real_images.shape[0], gan.latent_size], device=device)
                    fakes = gan.generator(latents)

                    fakes_score = gan.discriminator(fakes)
                    fakes_score_mean = -fakes_score.mean()
                    fakes_score_mean.backward()

                    optimizerGenerator.step()
                    running_loss_generator += fakes_score_mean
                    

                if i % 100 == 0:
                    print(f"Training example {i} / {len(trainloader)}. DiscLoss: {running_loss_discriminator:.2f}, GenLoss: {running_loss_generator:.2f}")
                    running_loss_discriminator = 0
                    running_loss_generator = 0
            if epoch % 5 == 0:
                visual_samples(gan, 100, device, trainloader, step=epoch)
        
        torch.save(gan.state_dict(), 'q3_gan_save.pth')

    dimensions = 100
        
    gan.eval()
    #3.1 Visual samples
    visual_samples(gan, dimensions, device, testloader)

    #3.2 Disentangled representation
    disentangled_representation(gan, dimensions, device, epsilon=10)

    #3.3 Interpolation
    interpolation(gan, dimensions, device)

    img_dir = "images/gan/fid"
    save_1000_images(img_dir)
    