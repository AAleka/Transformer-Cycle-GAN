import time
import gc
import torch
import math
import albumentations as A
import numpy as np

from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
# from torchsummary import summary
# from torchmetrics import PeakSignalNoiseRatio
from skimage import color
from skimage.metrics import structural_similarity

from Utils import load_checkpoint
from Dataset import ABDataset
from Model import Generator

gc.collect()
torch.cuda.empty_cache()

dataset_name = "EyeQ"  # "EyeQ" "Mendeley"

path = "Results"
checkpoint = "Results/genb.pth.tar"
save_path = f"Results/Testing {dataset_name}"
TEST_DIR = f"datasets/{dataset_name}/test"


def masking(a, b):
    l_top = l_bottom = 0
    a = a[0]
    b = b[0]

    for i in range(a.shape[1]):
        if torch.sum(a[:, i, :]) != 0:
            break
        l_top += 1

    for i in range(a.shape[1]):
        if torch.sum(a[:, a.shape[1] - i - 1, :]) != 0:
            break
        l_bottom += 1

    b[:, :l_top, :] = 0
    b[:, b.shape[1] - l_bottom:, :] = 0

    return a, b


def PSNR_SSIM(orig_img, gen_img):
    gray_orig_img = color.rgb2gray(orig_img)
    gray_gen_img = color.rgb2gray(gen_img)

    mse = np.mean((gray_orig_img - gray_gen_img) ** 2)
    if mse == 0:
        psnr = 100
    else:
        max_pixel = 1.0
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))

    ssim = structural_similarity(gray_orig_img, gray_gen_img, multichannel=False, data_range=1.0)

    return round(psnr, 3), round(ssim, 3)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gen = Generator().to(DEVICE)
# summary(gen, (3, 256, 256))

load_checkpoint(checkpoint, gen, None, None)

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ]
)

val_dataset = ABDataset(
    root_a=TEST_DIR, transform=transforms
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    pin_memory=True,
)

loop = tqdm(val_loader, leave=True)
psnr_values = []
ssim_values = []

start = time.time()

for idx, image in enumerate(loop):
    image = image.to(DEVICE)

    with torch.cuda.amp.autocast():
        gen_image = gen(image)
        image, gen_image = masking(image*0.5+0.5, gen_image*0.5+0.5)

        save_image(gen_image, f"{save_path}/{idx}_fake.png")
        save_image(image, f"{save_path}/{idx}_real.png")

        image = image.permute(1, 2, 0).detach().cpu().numpy()
        gen_image = gen_image.permute(1, 2, 0).detach().cpu().numpy()

        psnr_values.append(PSNR_SSIM(image, gen_image)[0])
        ssim_values.append(PSNR_SSIM(image, gen_image)[1])

end = time.time()

metrics = [
    round(sum(psnr_values) / len(val_loader), 3),
    round(sum(ssim_values) / len(val_loader), 3),
    round((end - start) / len(val_loader), 3)
]

f = open(f"{path}/Results {dataset_name}.txt", 'w')
f.write(f"Testing PSNR :{metrics[0]} dB\n")
f.write(f"Testing SSIM :{metrics[1]}\n")
f.write(f"Single image time: {metrics[2]} seconds\n")
