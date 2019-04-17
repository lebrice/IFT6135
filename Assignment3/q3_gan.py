
import torch
from torch import nn
from torch.optim import Adam
from typing import Tuple, Iterable, Dict, Callable, Any

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
        
        self.g_optimizer = Adam(self.generator.parameters(), lr=3e-4)
        self.d_optimizer = Adam(self.discriminator.parameters(), lr=3e-4)

        # TODO: Tune this hyperparameter
        self.d_steps_per_g_step: int = 3

        self._training_the_discriminator: bool = None

    def train_g(self) -> torch.FloatTensor:
        self.training_the_generator = True
        self.g_optimizer.zero_grad()
        loss = self.g_loss()
        loss.backward()
        self.g_optimizer.step()
        return loss

    def train_d(self, real_images_batch: torch.Tensor) -> torch.FloatTensor:
        self.training_the_discriminator = False
        self.d_optimizer.zero_grad()
        real_images_batch.requires_grad = True
        loss = self.d_loss(real_images_batch)
        loss.backward()
        self.d_optimizer.step()
        return loss


    def g_loss(self) -> torch.FloatTensor:
        assert self.batch_size is not None, "d_loss should have been called before g_loss"
        latents = torch.randn([self.batch_size, self.latent_size], requires_grad=True)
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
            create_graph=False,
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
        if self._training_the_discriminator != value:
            self._training_the_discriminator = value
            for p in self.discriminator.parameters():
                p.requires_grad = value
            for p in self.generator.parameters():
                p.requires_grad = value
    
    @property
    def training_the_generator(self) -> bool:
        return not self.training_the_discriminator
    
    @training_the_generator.setter
    def training_the_generator(self, value: bool) -> None:
        self.training_the_discriminator = not value


def random_interpolation(x, y):
    a = torch.rand_like(x)
    return a * x + (1-a) * y


def call_fns_every_n_steps_iter(source_iter: Iterable[Any], n_to_function_dict: Dict[int, Callable]) -> Iterable[Any]:
    """
    Convenience function for training loops, logging, etc.
    """
    for i, element in enumerate(source_iter):
        yield element
        for n, function in n_to_function_dict.items():
            if i % n == 0 and i != 0:
                # print("step", i, "calling", function.__name__)
                function()


def main(epochs=20):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    import torchvision
    from score_fid import get_test_loader
    from torch.optim import Adam
    
    gan = GAN()
    gan = gan.to(device)
    gan.train()

    running_d_loss = 0
    running_g_loss = 0

    svhn_loader = get_test_loader(64)
    num_batches = len(svhn_loader)

    for epoch in range(epochs):
        print(f"------- EPOCH {epoch} --------")
        
        image_gen = (images for images, _ in svhn_loader)

        def g_step():
            nonlocal running_g_loss
            loss = gan.train_g()            
            running_g_loss += loss

        def log_losses_and_reset():
            nonlocal running_d_loss, running_g_loss
            print(f"Training batch {i + 1} / {num_batches}. D-Loss: {running_d_loss:.2f}, G-Loss: {running_g_loss:.2f}")
            running_d_loss, running_g_loss = 0, 0

        n_to_callback_dict = {
            gan.d_steps_per_g_step: g_step,
            100: log_losses_and_reset,
        }

        i = 0
        for image_batch in call_fns_every_n_steps_iter(image_gen, n_to_callback_dict):
            d_loss = gan.train_d(image_batch)
            running_d_loss += d_loss
            i += 1

    torch.save(gan.state_dict(), 'q3_gan_save.pth')

    # Generate new images
    z = torch.randn(64, 100, device=device)
    generated = gan.generator(z)
    torchvision.utils.save_image(generated, 'gan-generated.png', normalize=True)
        
if __name__ == "__main__":
    main()