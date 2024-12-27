# training.py

import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, initialize_weights


num_epochs = 35
batch_size = 128
image_channels = 1
image_size=64
image_dimension = 64
z_dim  = 100
LEARNING_RATE = 2e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(image_channels)], [0.5 for _ in range(image_channels)]
        )
    ]
)


dataset = datasets.MNIST(root="data", download = True, train = "true", transform = transforms)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
gen = Generator(z_dim, image_channels, image_dimension).to(device)
disc = Discriminator(image_channels, image_dimension).to(device)
initialize_weights(gen)
initialize_weights(disc)

gen_optim = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
disc_optim = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()
fixed_noise = torch.randn(32, z_dim, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

for epoch in range(num_epochs):
    for batch, (X, y) in enumerate(dataloader):
        X= X.to(device)
        batch_size = X.shape[0]
        noise = torch.randn(batch_size, z_dim,1,1).to(device)
        fake = gen(noise)
        disc_real = disc(X).view(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real+loss_disc_fake)/2
        disc.zero_grad()
        loss_disc.backward(retain_graph = True)
        disc_optim.step()
        output = disc(fake).view(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        gen_optim.step()
        if batch == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch}/{len(dataloader)} \
                      Loss D: {loss_disc:.8f}, loss G: {loss_gen:.8f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(X[:32], normalize=True)
                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )

                step += 1
