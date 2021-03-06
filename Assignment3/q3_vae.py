import numpy as np
import torch
import torchvision
from torch import nn, optim
from classify_svhn import get_data_loader, get_data_loaderNoNormalize
from q3_gan import Generator

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

#Use the same architecture as the GAN. 
class Decoder(Generator):

    def forward(self, x):
        x = x.view(x.size(0), -1, 1, 1)
        x = self.convTranspose(x)
        self.activation = nn.Tanh()
        x = self.activation(x)
        return x

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

def visual_samples(vae, dimensions, device, svhn_loader):
    # Generate new images
    z = torch.randn(64, dimensions, device=device)
    generated = vae.decoder(z)
    torchvision.utils.save_image(generated, 'images/vae/3_1vae-generated.png', normalize=True)

    #Original image vs Reconstruction 
    x = next(iter(svhn_loader))[0]
    torchvision.utils.save_image(x, 'images/vae/3_1vae-initial.png', normalize=True)
    x = x.to(device)
    y, mu, logvar = vae(x)
    torchvision.utils.save_image(y, 'images/vae/3_1vae-restored.png', normalize=True)
    
def disentangled_representation(vae, dimensions, device, epsilon = 3):
    #Sample from prior p(z) which is a Std Normal
    z = torch.randn(dimensions, device=device)
    
    #Copy this tensor times its number of dimensions and make perturbations on each dimension
    #The first element is the original sample
    z = z.repeat(dimensions+1, 1)
    for i, sample in enumerate(z[1:]):
        sample[i] += epsilon

    generated = vae.decoder(z)
    torchvision.utils.save_image(generated[0], 'images/vae/3_2_original.png', normalize=True)
    torchvision.utils.save_image(generated[1:], 'images/vae/3_2positive_eps.png', nrow=10, normalize=True)

    #Do the same with the negative epsilon
    epsilon = -2*epsilon
    for i, sample in enumerate(z[1:]):
        sample[i] += epsilon

    #Make a batch of the pertubations and pass it through the decoder
    generated = vae.decoder(z)
    torchvision.utils.save_image(generated[1:], 'images/vae/3_2negative_eps.png', nrow=10, normalize=True)

def make_tensor(tens_list):
    batch = None
    for img in tens_list:
        if batch is None:
            batch = img
        else:
            batch = torch.cat((batch,img))
    return batch

def interpolation(vae, dimensions, device):
    # Interpolate in the latent space between z_0 and z_1
    z_0 = torch.randn(1,dimensions, device=device)
    z_1 = torch.randn(1,dimensions, device=device)
    generated = []

    for i in range(11):
        a = i/10
        z_a = a*z_0 + (1-a)*z_1
        output = vae.decoder(z_a)
        generated.append(output) 

    generated = torch.cat(generated)
    torchvision.utils.save_image(generated, 'images/vae/3_3latent.png', nrow=11, normalize=True)
    
    # Interpolate in the data space between x_0 and x_1
    x_0 = vae.decoder(z_0)
    x_1 = vae.decoder(z_1)
    x_a = torch.zeros(11,x_0.size()[1],x_0.size()[2],x_0.size()[3], device=device)

    for i in range(11):
        a = i/10
        x_a[i] = a*x_0 + (1-a)*x_1

    torchvision.utils.save_image(x_a, 'images/vae/3_3data.png', nrow=11, normalize=True)


def save_1000_images(img_dir: str):
    import os
    vae = VAE()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vae.load_state_dict(torch.load('q3_vae_save.pth', map_location=device))
    vae = vae.to(device)
    vae.eval()
    
    for p in vae.parameters():
        p.requires_grad = False
        os.makedirs(f"{img_dir}/img/", exist_ok=True)
    for i in range(10):
        print(i)
        latents = torch.randn(100, 100, device=device)
        images = vae.decoder(latents)
        for j, image in enumerate(images):
            filename = f"images/vae/fid/img/{i * 100 + j:03d}.png"
            torchvision.utils.save_image(image, filename, normalize=True)




if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    vae = VAE()
    vae = vae.to(device)

    optimizer = optim.Adam(vae.parameters(), lr=3e-4)

    running_loss = 0

    trainloader, validloader, testloader = get_data_loader("svhn", 64)
    
    try: 
        vae.load_state_dict(torch.load('q3_vae_save.pth', map_location=device))
        print('----Using saved model----')

    except FileNotFoundError:
        for epoch in range(5):

            print(f"------- EPOCH {epoch} --------")

            for i, (x, _) in enumerate(trainloader):
                vae.train()
                optimizer.zero_grad()

                x = x.to(device)
                y, mu, logvar = vae(x)

                loss = -ELBO(y, x, mu, logvar)
                running_loss += loss

                loss.backward()
                optimizer.step()

                if(i%10 == 0):
                    visual_samples(vae, 100, device, testloader)

                if (i + 1) % 100 == 0:
                    print(f"Training example {i + 1} / {len(trainloader)}. Loss: {running_loss}")
                    running_loss = 0

        torch.save(vae.state_dict(), 'q3_vae_save.pth')

    dimensions = 100
    
    #3.1 Visual samples
    visual_samples(vae, dimensions, device, testloader)

    #3.2 Disentangled representation
    disentangled_representation(vae, dimensions, device, epsilon=10)

    #3.3 Interpolation
    interpolation(vae, dimensions, device)
    
    img_dir = "images/vae/fid"
    save_1000_images(img_dir)
