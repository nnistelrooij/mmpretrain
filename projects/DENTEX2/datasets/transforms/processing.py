import math
from typing import Tuple

import numpy as np
from mmcv.transforms.utils import cache_randomness

from mmpretrain.registry import TRANSFORMS
from mmpretrain.datasets.transforms import RandomResizedCrop


@TRANSFORMS.register_module()
class RandomResizedClassPreservingCrop(RandomResizedCrop):

    @cache_randomness
    def rand_crop_params(self, img: np.ndarray) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (ndarray): Image to be cropped.

        Returns:
            tuple: Params (offset_h, offset_w, target_h, target_w) to be
                passed to `crop` for a random sized crop.
        """
        h, w = img.shape[:2]
        area = h * w

        mask_area = np.sum(img[..., 2] > 0)

        for _ in range(self.max_attempts):
            target_area = np.random.uniform(*self.crop_ratio_range) * area
            log_ratio = (math.log(self.aspect_ratio_range[0]),
                         math.log(self.aspect_ratio_range[1]))
            aspect_ratio = math.exp(np.random.uniform(*log_ratio))
            target_w = int(round(math.sqrt(target_area * aspect_ratio)))
            target_h = int(round(math.sqrt(target_area / aspect_ratio)))

            if (
                target_w <= 0 or target_w > w or
                target_h <= 0 or target_h > h
            ):
                continue

            offset_h = np.random.randint(0, h - target_h + 1)
            offset_w = np.random.randint(0, w - target_w + 1)

            slices = (
                slice(offset_h, offset_h + target_h),
                slice(offset_w, offset_w + target_w),
            )
            crop_mask_area = np.sum(img[slices][..., 2] > 0)

            if mask_area != crop_mask_area:
                continue
            
            return offset_h, offset_w, target_h, target_w

        # Fallback to central crop
        in_ratio = float(w) / float(h)
        if in_ratio < min(self.aspect_ratio_range):
            target_w = w
            target_h = int(round(target_w / min(self.aspect_ratio_range)))
        elif in_ratio > max(self.aspect_ratio_range):
            target_h = h
            target_w = int(round(target_h * max(self.aspect_ratio_range)))
        else:  # whole image
            target_w = w
            target_h = h
        offset_h = (h - target_h) // 2
        offset_w = (w - target_w) // 2
        return offset_h, offset_w, target_h, target_w
