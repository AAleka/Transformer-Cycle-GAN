import os

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
# from torchsummary import summary

from Utils import save_checkpoint, load_checkpoint
from Dataset import ABDataset
from Model import Generator, Discriminator

import gc
gc.collect()
torch.cuda.empty_cache()


def train_fn(disc_A, disc_B, gen_A, gen_B, loader, opt_disc, opt_gen, l1, mse,  d_scaler, g_scaler, epoch):
    global count
    avg_dloss = 0
    avg_gloss = 0
    loop = tqdm(loader, leave=True)
    for idx, (a, b) in enumerate(loop):
        a = a.to(DEVICE)
        b = b.to(DEVICE)

        with torch.cuda.amp.autocast():
            fake_a = gen_A(b)
            D_A_real = disc_A(a)
            D_A_fake = disc_A(fake_a.detach())
            D_A_real_loss = mse(D_A_real, torch.ones_like(D_A_real))
            D_A_fake_loss = mse(D_A_fake, torch.zeros_like(D_A_fake))
            D_A_loss = D_A_real_loss + D_A_fake_loss

            fake_b = gen_B(a)
            D_B_real = disc_B(b)
            D_B_fake = disc_B(fake_b.detach())
            D_B_real_loss = mse(D_B_real, torch.ones_like(D_B_real))
            D_B_fake_loss = mse(D_B_fake, torch.zeros_like(D_B_fake))
            D_B_loss = D_B_real_loss + D_B_fake_loss

            D_loss = (D_A_loss + D_B_loss)/2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        with torch.cuda.amp.autocast():
            D_A_fake = disc_A(fake_a)
            D_B_fake = disc_B(fake_b)
            loss_G_A = mse(D_A_fake, torch.ones_like(D_A_fake))
            loss_G_B = mse(D_B_fake, torch.ones_like(D_B_fake))

            cycle_b = gen_B(fake_a)
            cycle_a = gen_A(fake_b)
            cycle_b_loss = l1(b, cycle_b)
            cycle_a_loss = l1(a, cycle_a)

            identity_b = gen_B(b)
            identity_a = gen_A(a)
            identity_b_loss = l1(b, identity_b)
            identity_a_loss = l1(a, identity_a)

            G_loss = (
                loss_G_B
                + loss_G_A
                + cycle_b_loss * LAMBDA_CYCLE
                + cycle_a_loss * LAMBDA_CYCLE
                + identity_a_loss * LAMBDA_IDENTITY
                + identity_b_loss * LAMBDA_IDENTITY
            )

            avg_dloss += D_loss.item()
            avg_gloss += G_loss.item()

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 100 == 0:
            save_image(fake_a*0.5+0.5, f"{path}/Generated from HQ/{count}_fake.png")
            save_image(fake_b*0.5+0.5, f"{path}/Generated from LQ/{count}_fake.png")
            save_image(b*0.5+0.5, f"{path}/Generated from HQ/{count}_real.png")
            save_image(a*0.5+0.5, f"{path}/Generated from LQ/{count}_real.png")
            count += 1
        loop.set_postfix(epoch=epoch+1, loss_g=avg_gloss/(idx+1), loss_d=avg_dloss/(idx+1))


def main():
    disc_A = Discriminator().to(DEVICE)
    disc_B = Discriminator().to(DEVICE)
    gen_A = Generator(width=IMAGE_WIDTH, height=IMAGE_HEIGHT).to(DEVICE)
    gen_B = Generator(width=IMAGE_WIDTH, height=IMAGE_HEIGHT).to(DEVICE)

    opt_disc = optim.Adam(
        list(disc_A.parameters()) + list(disc_B.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_A.parameters()) + list(gen_B.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if LOAD_MODEL:
        load_checkpoint(
            CHECKPOINT_GEN_A, gen_A, opt_gen, LEARNING_RATE,
        )
        load_checkpoint(
            CHECKPOINT_GEN_B, gen_B, opt_gen, LEARNING_RATE,
        )
        load_checkpoint(
            CHECKPOINT_DISC_A, disc_A, opt_disc, LEARNING_RATE,
        )
        load_checkpoint(
            CHECKPOINT_DISC_B, disc_B, opt_disc, LEARNING_RATE,
        )

    dataset = ABDataset(
        root_a=TRAIN_DIR+"/LQ", root_b=TRAIN_DIR+"/HQ", transform=transforms
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(disc_A, disc_B, gen_A, gen_B, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler, epoch)

        if SAVE_MODEL:
            save_checkpoint(gen_A, opt_gen, filename=CHECKPOINT_GEN_A)
            save_checkpoint(gen_B, opt_gen, filename=CHECKPOINT_GEN_B)
            save_checkpoint(disc_A, opt_disc, filename=CHECKPOINT_DISC_A)
            save_checkpoint(disc_B, opt_disc, filename=CHECKPOINT_DISC_B)


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAIN_DIR = "datasets/EyeQ/train"
    path = "Results"
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-5
    LAMBDA_IDENTITY = 10
    LAMBDA_CYCLE = 10
    NUM_WORKERS = 4
    NUM_EPOCHS = 500
    LOAD_MODEL = False
    SAVE_MODEL = True
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256
    CHECKPOINT_GEN_A = f"{path}/gena.pth.tar"
    CHECKPOINT_GEN_B = f"{path}/genb.pth.tar"
    CHECKPOINT_DISC_A = f"{path}/disca.pth.tar"
    CHECKPOINT_DISC_B = f"{path}/discb.pth.tar"
    count = 0

    if not os.path.exists("Results"):
        os.mkdir("Results")
        os.mkdir("Results/Generated from HQ")
        os.mkdir("Results/Generated from LQ")

    transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
            ToTensorV2(),
         ],
        additional_targets={"image0": "image"},
    )

    main()
