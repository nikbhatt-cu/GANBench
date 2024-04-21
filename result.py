class result:
    def __init__(self, real_path, mask_path, gen_path, lpips, ssim, overlay_path, animated_gif_path,
                 percentage_masked, mask_size):
        self.real_path = real_path
        self.mask_path = mask_path
        self.gen_path = gen_path
        self.lpips = lpips
        self.ssim = ssim
        self.overlay_path = overlay_path
        self.animated_gif_path = animated_gif_path
        self.percentage_masked = percentage_masked
        self.mask_size = mask_size


# appending instances to list
#list.append(geeks('Akash', 2))
