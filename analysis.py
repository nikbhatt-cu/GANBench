import os
from glob import glob
import argparse
from pathlib import Path
import lpips
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageOps
import torch
import cv2
import csv
from typing import List

from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from pytorch_msssim import ssim

import adapter
import result

# hack to disable MPO reading
from PIL import JpegImagePlugin
JpegImagePlugin._getmp = lambda x: None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, help="For display purposes")
    parser.add_argument("--real-dir", type=Path, help="Real dataset directory path.")
    parser.add_argument("--mask-dir", type=Path, help="Pre-generated mask directory path.")
    parser.add_argument("--generated-dir", type=Path, help="Generated image directory path.")
    parser.add_argument("--result-dir", type=Path, help="result directory", required=False)
    parser.add_argument("--device", type=str, help="Device to use.")
    return parser.parse_args()


def load_and_resize_image(image_path: str, needs_scaling, resolution):
    image = Image.open(image_path)
    if needs_scaling is True and image.size != resolution:
        image = image.resize(resolution, Image.BICUBIC)
    return image


def convert_image_to_tensor(image: Image):
    img = np.array(image)
    img = torch.Tensor(img).float() / 255.0
    img = img.permute(2, 0, 1)
    return img


def tint_mask(image_path: str, mask_is_black):
    image = Image.open(image_path).convert("RGB")
    if mask_is_black is False:
        image = ImageOps.invert(image)

    image_data = image.getdata()
    new_image_list = []
    for pixel in image_data:
        # change black to light blue
        if pixel[0] == 0:
            new_image_list.append((0, 0, 255, 255))
        else:
            new_image_list.append((0, 0, 0, 0))

    # update image data
    image.putdata(new_image_list)
    return image


def make_alpha_mask_image(img: Image, mask_is_black):
    image = img.copy()

    if mask_is_black is False:
        image = ImageOps.invert(img)

    image_data = image.getdata()
    new_image_list = []
    for pixel in image_data:
        # change black to light blue
        if pixel[0] == 0:
            new_image_list.append((0, 60, 255, 100))
        else:
            new_image_list.append((0, 0, 0, 0))

    # update image data
    image.putdata(new_image_list)
    return image


def make_animated_gif(first: Image, other_list: list, output_path: str):
    first.save(output_path, save_all=True, append_images=other_list, duration=500)


def composite_mask(base_image: Image, tinted_mask_image: Image, mask_path: str, mask_is_black):
    base_img = base_image
    if base_img.mode != "RGBA":
        base_img = base_img.convert("RGBA")
    mask_image = Image.open(mask_path).convert("RGBA")
    mask_image = make_alpha_mask_image(mask_image, mask_is_black)

    if mask_is_black is False:
        mask_image = ImageOps.invert(mask_image)

    composite_image = Image.composite(tinted_mask_image, base_img, mask_image)
    return composite_image


# from MI-GAN: https://github.com/Picsart-AI-Research/MI-GAN
def get_fid_activations(model, imgs):
    with torch.no_grad():
        activations = model(imgs)[0]
    if activations.size(2) != 1 or activations.size(3) != 1:
        activations = adaptive_avg_pool2d(activations, output_size=(1, 1))
    activations = activations.squeeze(3).squeeze(2).cpu().numpy()
    return activations


def calculate_fid(real_activations_array, fake_activations_array):
    mu_real = np.mean(real_activations_array, axis=0)
    sigma_real = np.cov(real_activations_array, rowvar=False)
    mu_fake = np.mean(fake_activations_array, axis=0)
    sigma_fake = np.cov(fake_activations_array, rowvar=False)
    fid_value = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    return fid_value


def calculate_mask_percentage(image_path: Path, mask_is_black: bool):
    image = cv2.imread(str(image_path), 0)
    count = cv2.countNonZero(image)
    total_pixels = image.size
    fraction = count / total_pixels
    if mask_is_black:
        fraction = 1.0 - fraction

    return fraction * 100.0


def write_csv(result_array, output_dir, lpips_mean, lpips_std, lpips_var, ssim_mean, ssim_std,
              ssim_var, fid):
    output_file = Path(output_dir).joinpath("result.csv")
    with open(str(output_file), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['LPIPS Mean', 'LPIPS Std Dev', 'LPIPS Variance', 'SSIM Mean', 'SSIM Std Dev', 'SSIM Variance',
                         'FID'])
        writer.writerow([lpips_mean, lpips_std, lpips_var, ssim_mean, ssim_std, ssim_var, fid])
        writer.writerow(['Original Path', 'Generated Path', 'Mask Path', 'Overlay Path', 'Animated GIF Path', 'LPIPS',
                         'SSIM', 'Percentage Masked', 'Mask Size H', 'Mask Size V'])
        for result_entry in result_array:
            row = [result_entry.real_path, result_entry.gen_path, result_entry.mask_path, result_entry.overlay_path,
                   result_entry.animated_gif_path, result_entry.lpips, result_entry.ssim, result_entry.percentage_masked,
                   result_entry.mask_size[0], result_entry.mask_size[1]]
            writer.writerow(row)


def run():
    args = get_args()

    args.real_dir

    # can replace these with hard-coded paths to ease debugging, but if you pass them on the command line, those
    # take precedence.
    model_name = args.model_name if args.model_name is not None else "HiFill (Places)"
    real_dir = args.real_dir if args.real_dir is not None else Path("../GANBenchOut/Places/final-validation")
    mask_dir = args.mask_dir if args.mask_dir is not None else Path("../GANBenchOut/Places/final-validation-masks")
    gen_dir = args.generated_dir if args.generated_dir is not None else Path("../GANBenchOut/Results/Places/HiFill")
    result_dir = args.result_dir if args.result_dir is not None else Path("../GANBenchOut/Analysis/Places/HiFill")
    device = args.device if args.device is not None else "cpu"
    '''
        python main.py \
        --model-name "MI-GAN (Faces-256)" \
        --real-dir ../FFHQ/1024-100 \
        --mask-dir ../MI-GAN/MI-GAN-main/ffhq256_mask \
        --generated-dir ../MI-GAN/MI-GAN-main/output \
        --result-dir ./results \
        --device cpu
    '''

    # make result directory for our data
    os.makedirs(str(result_dir), exist_ok=True)

    # load LPIPS
    compute_lpips = lpips.LPIPS(net="alex")

    # load Inception for FID
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception_v3 = InceptionV3([block_idx]).to(device)
    inception_v3.eval()

    # use an adapter to map the "real" image's path to mask and result paths
    the_adapter = adapter.HiFillOpenImages512Adapter()
    mask_is_black = the_adapter.mask_is_black()

    # list of result objects
    result_list = []

    # the real_dir drives the iteration. Find all the image files in the directory
    img_extensions = the_adapter.img_extensions()
    img_paths = []
    for img_extension in img_extensions:
        img_paths += glob(os.path.join(real_dir, "**", f"*{img_extension}"), recursive=True)

    img_paths = sorted(img_paths)

    lpips_list = []
    ssim_list = []

    # needed for FID
    real_activations_array = np.empty((len(img_paths), 2048))
    fake_activations_array = np.empty((len(img_paths), 2048))

    img_index = 0

    # iterate through real images
    for img_path in tqdm(img_paths):
        image_name = Path(img_path).name
        mask_path, gen_path = the_adapter(image_name, mask_dir, gen_dir)
        # print(f"{image_name}: mask_path = {mask_path}, gen_path = {gen_path}")

        real_img_loaded = load_and_resize_image(img_path, the_adapter.needs_scaling(), the_adapter.resolution())
        real_img = convert_image_to_tensor(real_img_loaded)

        gen_img_loaded = load_and_resize_image(gen_path, the_adapter.needs_scaling(), the_adapter.resolution())
        gen_img = convert_image_to_tensor(gen_img_loaded)

        real_imgs = torch.stack([real_img], dim=0)
        gen_imgs = torch.stack([gen_img], dim=0)

        # fid calculation update
        real_activations_array[img_index] = get_fid_activations(inception_v3, real_imgs)
        fake_activations_array[img_index] = get_fid_activations(inception_v3, gen_imgs)

        # compute LPIPS
        with torch.no_grad():
            lpips_output = compute_lpips(real_imgs, gen_imgs, normalize=True)
            lpips_val = lpips_output.detach().cpu().numpy()[0][0][0][0]
            lpips_list.append(lpips_val)

        # compute SSIM
        ssim_val = ssim(real_imgs, gen_imgs, data_range=1, size_average=False)  # return (N,)
        ssim_val = ssim_val.numpy()[0]
        ssim_list.append(ssim_val)

        # generate mask overlay image and store on disk
        tinted_mask_image = tint_mask(mask_path, mask_is_black)
        mask_size = tinted_mask_image.size
        overlay_name = the_adapter.overlay_name(image_name, "png")
        overlay_path = Path.joinpath(result_dir, overlay_name)
        composite_image = composite_mask(real_img_loaded, tinted_mask_image, mask_path, mask_is_black)
        composite_image.save(str(overlay_path))

        # generate animated GIF and store on disk
        animated_gif_name = the_adapter.animated_gif(img_path, "gif")
        animated_gif_path = Path.joinpath(result_dir, animated_gif_name)
        make_animated_gif(real_img_loaded, [composite_image, gen_img_loaded, real_img_loaded, gen_img_loaded],
                                            animated_gif_path)

        percentMasked = calculate_mask_percentage(mask_path, mask_is_black)

        # create result object and store inside
        result_object = result.result(Path(img_path).resolve(), mask_path.resolve(), gen_path.resolve(),
                                      lpips_val, ssim_val, overlay_path.resolve(),
                                      animated_gif_path.resolve(), percentMasked, mask_size)
        result_list.append(result_object)

        img_index += 1

    lpips_mean = np.mean(lpips_list)
    lpips_stddev = np.std(lpips_list)
    lpips_variance = np.var(lpips_list)
    ssim_mean = np.mean(ssim_list)
    ssim_stddev = np.std(ssim_list)
    ssim_variance = np.var(ssim_list)

    print(f"{model_name}: lpips mean: {np.median(lpips_list)}")
    print(f"{model_name}: ssim mean: {np.median(ssim_list)}")

    # after the processing is done, then I can calculate FID, as long as there are at least 2048 images (FID rule)
    if len(img_paths) >= 2048:
        fid_score = calculate_fid(real_activations_array, fake_activations_array)
        print(f"{model_name}: FID score: {fid_score}")
    else:
        fid_score = -1
        print("unable to calculate FID - at least 2048 images are required.")

    write_csv(result_list, result_dir, lpips_mean, lpips_stddev, lpips_variance,
              ssim_mean, ssim_stddev, ssim_variance, fid_score)
    print(f"{model_name}: got {len(result_list)} result objects. Output written to {result_dir}")


if __name__ == "__main__":
    run()
