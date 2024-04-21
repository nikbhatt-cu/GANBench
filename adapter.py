import os
from pathlib import Path


class Adapter:
    def __init__(self):
        pass

    # return mask path and result path
    # customize this to match your own naming conventions
    def __call__(self, real_image_name, mask_path_str, result_path_str):
        stem = Path(real_image_name).stem
        mask_image_path = mask_path_str / f"{stem}.png"
        result_image_path = result_path_str / f"{stem}.png"
        return mask_image_path, result_image_path


    # what should the overlay image name be?
    def overlay_name(self, real_image_name, path_extension):
        stem = Path(real_image_name).stem
        return f"{stem}_overlay.{path_extension}"


    # what should the animated GIF name be?
    def animated_gif(self, real_image_name, path_extension):
        stem = Path(real_image_name).stem
        return f"{stem}_overlay.{path_extension}"


    # what is the resolution of the generated images so the real images can be scaled
    def resolution(self):
        return 256, 256

    # if this is true, then resolution will checked and images conformed to that size.
    def needs_scaling(self):
        return True


    # what file extensions are the real images?
    def img_extensions(self):
        return {".jpg", ".jpeg", ".png"}


    # what file extensions are the real images?
    def mask_is_black(self):
        return True


class Migan256Adapter(Adapter):
    def __init__(self):
        super().__init__()

    # return mask path and result path
    # customize this to match your own naming conventions
    def __call__(self, real_image_name, mask_path_str, result_path_str):
        stem = Path(real_image_name).stem
        mask_image_path = mask_path_str / f"img000{stem}.png"
        result_image_path = result_path_str / f"{stem}.png"
        return mask_image_path, result_image_path


class Migan512Adapter(Adapter):
    def __init__(self):
        super().__init__()

    # return mask path and result path
    # customize this to match your own naming conventions
    def __call__(self, real_image_name, mask_path_str, result_path_str):
        stem = Path(real_image_name).stem
        mask_image_path = mask_path_str / f"img000{stem}.png"
        result_image_path = result_path_str / f"{stem}.png"
        return mask_image_path, result_image_path

    def resolution(self):
        return 512, 512


class HiFillAdapter(Adapter):
    def __init__(self):
        super().__init__()

    def needs_scaling(self):
        return False

    # return mask path and result path
    # customize this to match your own naming conventions
    def __call__(self, real_image_name, mask_path_str, result_path_str):
        stem = Path(real_image_name).stem
        mask_image_path = mask_path_str / f"{stem}.jpg"
        result_image_path = result_path_str / f"{real_image_name}_inpainted.jpg"
        return mask_image_path, result_image_path


class HiFillAdapterPlaces(Adapter):
    def __init__(self):
        super().__init__()

    def needs_scaling(self):
        return True

    def resolution(self):
        return 512, 512

    # return mask path and result path
    # customize this to match your own naming conventions
    def __call__(self, real_image_name, mask_path_str, result_path_str):
        stem = Path(real_image_name).stem
        mask_image_path = mask_path_str / f"{stem}.png"
        result_image_path = result_path_str / f"{real_image_name}_inpainted.jpg"
        return mask_image_path, result_image_path


class MiganPlaces512Adapter(Adapter):
    def __init__(self):
        super().__init__()

    # return mask path and result path
    # customize this to match your own naming conventions
    def __call__(self, real_image_name, mask_path_str, result_path_str):
        stem = Path(real_image_name).stem
        mask_image_path = mask_path_str / f"{stem}.png"
        result_image_path = result_path_str / f"{stem}.png"
        return mask_image_path, result_image_path

    def resolution(self):
        return 512, 512


class HiFillOpenImages512Adapter(Adapter):
    def __init__(self):
        super().__init__()

    # return mask path and result path
    # customize this to match your own naming conventions
    def __call__(self, real_image_name, mask_path_str, result_path_str):
        stem = Path(real_image_name).stem
        mask_image_path = mask_path_str / f"{stem}.png"
        result_image_path = result_path_str / f"{real_image_name}_inpainted.jpg"
        return mask_image_path, result_image_path

    def resolution(self):
        return 512, 512



