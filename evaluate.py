import math
import cv2
import glob
import numpy as np

from skimage import color
from skimage.metrics import structural_similarity


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


dataset = "Mendeley"  # "EyeQ" "Mendeley"
method = "Domain"  # "CLAHE" "CycleCBAM" "CycleGAN" "Domain" "DPFR" "TransCycleGAN"

reals = glob.glob(f"Qualitative/{method}/Testing {dataset}/*_real.png")
fakes = glob.glob(f"Qualitative/{method}/Testing {dataset}/*_fake.png")

psnr_values = []
ssim_values = []

for (real, fake) in zip(reals, fakes):
    img_real = cv2.imread(real) / 255.0
    img_fake = cv2.imread(fake) / 255.0

    psnr_values.append(PSNR_SSIM(img_real, img_fake)[0])
    ssim_values.append(PSNR_SSIM(img_real, img_fake)[1])

metrics = [
    round(sum(psnr_values) / len(reals), 3),
    round(sum(ssim_values) / len(reals), 3)
]

f = open(f"Qualitative/{method}/Results {dataset}.txt", 'w')
f.write(f"Testing PSNR :{metrics[0]} dB\n")
f.write(f"Testing SSIM :{metrics[1]}\n")
