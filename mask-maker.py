import os
import shutil
import random
from glob import glob
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageOps
import torch

import adapter

# hack to disable MPO reading
from PIL import JpegImagePlugin
JpegImagePlugin._getmp = lambda x: None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, help="Location of source photos")
    parser.add_argument("--mask-dir", type=Path, help="Location of source masks")
    parser.add_argument("--final-source-dir", type=Path, help="Where to write final source photos")
    parser.add_argument("--final-mask-dir", type=Path, help="Where to write final masks")
    parser.add_argument("--final-composite-dir", type=Path, help="Where to write composite images (optional)")
    parser.add_argument("--match-mask", type=bool, help="Assign mask based on source file name (vs. random)")
    parser.add_argument("--mask-is-black", type=bool, help="False the mask pixels are white")
    parser.add_argument("--verbose", type=bool, help="True for more logging")
    return parser.parse_args()


def load_and_resize_image(image_path: str, needs_scaling, resolution):
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    if needs_scaling is True and image.size != resolution:
        image = image.resize(resolution, Image.BICUBIC)
    return image


def convert_image_to_tensor(image: Image):
    img = np.array(image)
    img = torch.Tensor(img).float() / 255.0
    img = img.permute(2, 0, 1)
    return img


def tint_mask(image: Image, mask_is_black):
    image = image.convert("RGB")
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


def make_tinted_alpha_mask_image(img: Image, mask_is_black):
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


def make_alpha_mask_image(img: Image, mask_is_black):
    image = img.copy()

    if mask_is_black is False:
        image = ImageOps.invert(img)

    image_data = image.getdata()
    new_image_list = []
    for pixel in image_data:
        # change black to light blue
        if pixel[0] == 0:
            new_image_list.append((255, 255, 255, 255))
        else:
            new_image_list.append((0, 0, 0, 0))

    # update image data
    image.putdata(new_image_list)
    return image


def composite_tinted_mask(base_image: Image, tinted_mask_image: Image, mask_image: Image):
    base_img = base_image
    if base_img.mode != "RGBA":
        base_img = base_img.convert("RGBA")
    if mask_image.mode != "RGBA":
        mask_image = mask_image.convert("RGBA")
    mask_image = make_tinted_alpha_mask_image(mask_image, True)

    composite_image = Image.composite(tinted_mask_image, base_img, mask_image)
    return composite_image


def composite_mask(base_image: Image, mask_image: Image):
    base_img = base_image
    if base_img.mode != "RGBA":
        base_img = base_img.convert("RGBA")
    if mask_image.mode != "RGBA":
        mask_image = mask_image.convert("RGBA")
    alpha_mask_image = make_alpha_mask_image(mask_image, True)
    composite_image = Image.composite(mask_image, base_img, alpha_mask_image)
    return composite_image


def make_image_list(source_dir, the_adapter):
    # make a list of all input photos
    img_extensions = the_adapter.img_extensions()
    img_paths = []
    for img_extension in img_extensions:
        img_paths += glob(os.path.join(source_dir, "**", f"*{img_extension}"), recursive=True)

    img_paths = sorted(img_paths)
    return img_paths


def make_mask_dict(mask_dir):
    mask_paths = []
    mask_extension = ".png"
    mask_paths += glob(os.path.join(mask_dir, "**", f"*{mask_extension}"), recursive=True)

    # for each path, have to figure out if it is one of many
    # need to make a dictionary because sometimes there are multiple mask PNGs for a single image
    # key = mask_root, value = list of full paths
    mask_path_dict = dict()
    for mask_path in mask_paths:
        mask_pieces = mask_path.split("_")
        mask_root = Path(mask_pieces[0]).stem
        existing = mask_path_dict.get(mask_root)
        if existing is None:
            existing = [mask_path]
        else:
            existing += [mask_path]
        mask_path_dict[mask_root] = existing
    return mask_path_dict


def make_final_list(img_paths, match_mask, mask_path_dict, output_resolution, output_dir: Path, verbose):
    final_image_paths = list()
    for img_path in tqdm(img_paths, desc="Source Photos"):
        true_path = Path(img_path)
        stem = true_path.stem
        pathExtension = true_path.suffix
        doCopy = True

        # if matching masks then only copy if we have a mask
        if match_mask:
            doCopy = (mask_path_dict.get(stem) is not None)
            if doCopy is False:
                if verbose: print(f"Not copying {stem} because no mask was found")
            else:
                if verbose: print(f"Copying {stem} because mask was found")

        if doCopy:
            leaf_name = stem + pathExtension
            output_path = str(output_dir.joinpath(leaf_name))
            if output_resolution is not None:
                image = load_and_resize_image(img_path, True, output_resolution)
                image.save(output_path)
            else:
                shutil.copy(img_path, output_path)
            final_image_paths += [output_path]

    return final_image_paths


def write_composite_scale_mask(original_img_path, mask_list, mask_is_black, mask_output_dir:Path):
    mask_file_name = mask_list[0]
    mask_stem = Path(original_img_path).stem
    mask_output_name = mask_output_dir.joinpath(mask_stem + ".png")

    # shortcut if there is only one item.
    if len(mask_list) == 1 and mask_is_black is True and output_size is None:
        shutil.copy(mask_file_name, str(mask_output_name))
    else:
        final_mask_image = None

        image = Image.open(original_img_path)
        image_size = image.size

        # load each mask, resize and invert as needed (must match the size of the original)
        for mask_file_name in mask_list:
            mask_image = load_and_resize_image(mask_file_name, True, image_size)
            if mask_is_black is False:
                mask_image = ImageOps.invert(mask_image)
            if final_mask_image is None:
                final_mask_image = mask_image
            else:
                final_mask_image = composite_mask(final_mask_image, mask_image)

        final_mask_image = final_mask_image.convert("RGB")
        final_mask_image.save(str(mask_output_name))

    return str(mask_output_name)


def make_final_masks(img_paths, match_mask, mask_path_dict, output_resolution, output_dir: Path,
                     mask_is_black, final_composite_dir, verbose):
    mask_path_list = list(mask_path_dict.values())
    for img_path in tqdm(img_paths, desc="Making Masks"):
        true_path = Path(img_path)
        stem = true_path.stem

        # if matching masks then only copy if we have a mask
        if match_mask:
            mask_entry_list = mask_path_dict.get(stem)
            if mask_entry_list is None:
                print(f"FAILURE: there must a mask entry for {stem}")
                return None
            else:
                mask_file_name = write_composite_scale_mask(img_path, mask_entry_list, mask_is_black, output_dir)
        else:
            # random selection
            mask_index = random.randint(0, len(mask_path_list) - 1)
            mask_entry_list = mask_path_list[mask_index]
            mask_file_name = write_composite_scale_mask(img_path, mask_entry_list, mask_is_black, output_dir)

        # if composite of the original and mask is desired, do that here.
        if final_composite_dir is not None:
            original_image = Image.open(img_path)
            mask_image = Image.open(mask_file_name)
            tinted_mask_image = tint_mask(mask_image, True)
            composite_image = composite_tinted_mask(original_image, tinted_mask_image, mask_image)
            img_leaf_ext = Path(img_path).stem + ".jpg"
            composite_image = composite_image.convert("RGB")
            composite_image.save(str(final_composite_dir.joinpath(img_leaf_ext)))


def run():
    args = get_args()

    # can replace these with hard-coded paths to ease debugging, but if you pass them on the command line, those
    # take precedence.
    source_dir = args.source_dir if args.source_dir is not None else Path("../OpenImage/validation")
    # source_dir = args.source_dir if args.source_dir is not None else Path("../MI-GAN/MI-GAN-main/examples/places2_512_freeform/images")
    mask_dir = args.mask_dir if args.mask_dir is not None else Path("../OpenImage/validation-masks")
    final_source_dir = args.final_source_dir if args.final_source_dir is not None else Path("../GANBenchOut/OpenImage/final-validation")
    final_mask_dir = args.final_mask_dir if args.final_mask_dir is not None else Path("../GANBenchOut/OpenImage/final-validation-masks")
    final_composite_dir = args.final_composite_dir if args.final_composite_dir is not None else Path("../GANBenchOut/OpenImage/final-validation-composite")
    match_mask = True
    mask_is_black = False
    verbose = False

    '''
    '''

    # make result directory for our data
    os.makedirs(str(final_source_dir), exist_ok=True)
    os.makedirs(str(final_mask_dir), exist_ok=True)
    os.makedirs(str(final_composite_dir), exist_ok=True)

    # supply an adapter for file system access
    the_adapter = adapter.Migan512Adapter()

    # make a list of all input photos and masks
    img_paths = make_image_list(source_dir, the_adapter)
    mask_path_dict = make_mask_dict(mask_dir)

    # make the final list of images (based on match_mask). Scale if needed
    resolution = the_adapter.resolution() if the_adapter.needs_scaling() else None
    img_paths = make_final_list(img_paths, match_mask, mask_path_dict, resolution, final_source_dir, verbose)

    # create final mask images and composite images (if desired)
    make_final_masks(img_paths, match_mask, mask_path_dict, resolution, final_mask_dir,
                     mask_is_black, final_composite_dir, verbose)


if __name__ == "__main__":
    run()
