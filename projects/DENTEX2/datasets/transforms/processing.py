import math
from typing import Any, Dict, Tuple

from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms import (
    BrightnessTransform,
    BrightnessMultiplicativeTransform,
    ContrastAugmentationTransform, 
    GammaTransform,
)
from batchgenerators.transforms.noise_transforms import (    
    GaussianBlurTransform,
    GaussianNoiseTransform,
)
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness
import numpy as np
from scipy import ndimage

from mmpretrain.registry import TRANSFORMS
from mmpretrain.datasets.transforms import RandomResizedCrop


@TRANSFORMS.register_module()
class RandomResizedClassPreservingCrop(RandomResizedCrop):

    def __init__(
        self,
        margin: float=0.02,
        two_pic: bool=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.margin = margin
        self.two_pic = two_pic

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
        margins = self.margin * h, self.margin * w

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
            tooth_slices = ndimage.find_objects(img[slices][..., 2] > 0)
            if self.margin > 0 and ( 
                not tooth_slices or
                tooth_slices[0][0].start < margins[0] or
                tooth_slices[0][1].start < margins[1] or
                (slices[0].stop - slices[0].start) - tooth_slices[0][0].stop < margins[0] or
                (slices[1].stop - slices[1].start) - tooth_slices[0][1].stop < margins[1]
            ):
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
    
    def transform(self, results: dict) -> dict:
        results = super().transform(results)

        if self.two_pic:
            results['img'] = [results['img'].copy(), results['img'].copy()]

        return results
 

@TRANSFORMS.register_module()
class NNUNetSpatialIntensityAugmentations(BaseTransform):

    def __init__(
        self,
        params: Dict[str, Any]={
            "do_elastic": False,
            "elastic_deform_alpha": (0., 900.),
            "elastic_deform_sigma": (9., 13.),
            "p_eldef": 0.2,
            "do_scaling": True,
            "scale_range": (0.7, 1.4),
            "independent_scale_factor_for_each_axis": False,
            "p_scale": 0.2,
            "do_rotation": True,
            "rotation_x": (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
            "rotation_y": (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
            "rotation_z": (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
            "rotation_p_per_axis": 1,
            "p_rot": 0.2,
            "random_crop": False,
            "random_crop_dist_to_border": None,
            "do_gamma": True,
            "gamma_retain_stats": True,
            "gamma_range": (0.7, 1.5),
            "p_gamma": 0.3,
            "do_mirror": True,
            "mirror_axes": (0, 1),
            "border_mode_data": "constant",
            "do_additive_brightness": False,
            "additive_brightness_p_per_sample": 0.15,
            "additive_brightness_p_per_channel": 0.5,
            "additive_brightness_mu": 0.0,
            "additive_brightness_sigma": 0.1,
        },
        patch_size=None,
        border_val_seg=-1,
        order_seg=1,
        order_data=3,
        disable=False,
        max_attempts=5,
        margin: float=0.05,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # ----------------------------------------------------------------------------------------------------------------------------------------------------------------
        # initialize list for train-time transforms
        tr_transforms = []
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not disable:
            # morphological spatial transforms
            tr_transforms.append(SpatialTransform(
                patch_size,
                patch_center_dist_from_border=None,
                do_elastic_deform=params.get("do_elastic"),
                alpha=params.get("elastic_deform_alpha"),
                sigma=params.get("elastic_deform_sigma"),
                do_rotation=params.get("do_rotation"),
                angle_x=params.get("rotation_x"),
                angle_y=params.get("rotation_y"),
                angle_z=params.get("rotation_z"),
                p_rot_per_axis=params.get("rotation_p_per_axis"),
                do_scale=params.get("do_scaling"),
                scale=params.get("scale_range"),
                border_mode_data=params.get("border_mode_data"),
                border_cval_data=0,
                order_data=order_data,
                border_mode_seg="constant",
                border_cval_seg=border_val_seg,
                order_seg=order_seg,
                random_crop=params.get("random_crop"),
                p_el_per_sample=params.get("p_eldef"),
                p_scale_per_sample=params.get("p_scale"),
                p_rot_per_sample=params.get("p_rot"),
                independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis"),
            ))
            # -------------------------------------------------------------------------------------------------------------------------------------------------------------
            # intensity transforms
            tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
            tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2, p_per_channel=0.5))
            tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))

            if params.get("do_additive_brightness"):
                tr_transforms.append(BrightnessTransform(
                    params.get("additive_brightness_mu"),
                    params.get("additive_brightness_sigma"), True,
                    p_per_sample=params.get("additive_brightness_p_per_sample"),
                    p_per_channel=params.get("additive_brightness_p_per_channel"),
                ))

            tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
            tr_transforms.append(SimulateLowResolutionTransform(
                zoom_range=(0.5, 1), per_channel=True, p_per_channel=0.5,
                order_downsample=0,  order_upsample=3, p_per_sample=0.25,
                ignore_axes=None,
            ))
            tr_transforms.append(GammaTransform(
                params.get("gamma_range"), True, True,
                retain_stats=params.get("gamma_retain_stats"),
                p_per_sample=0.1,
            ))  # inverted gamma

            if params.get("do_gamma"):
                tr_transforms.append(GammaTransform(
                    params.get("gamma_range"), False, True,
                    retain_stats=params.get("gamma_retain_stats"),
                    p_per_sample=params["p_gamma"],
                ))
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.transforms = Compose(tr_transforms)
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------------

        self.max_attempts = max_attempts
        self.margin = margin

    def transform(self, results: dict) -> dict:
        h, w = results['img_shape'][:2]
        margins = self.margin * h, self.margin * w

        for _ in range(self.max_attempts):
            img = results['img'].copy()

            results['data'] = img[None, None, ..., 0]
            results['seg'] = img[None, None, ..., 2] // 255
            results = self.transforms(**results)
            data = results.pop('data')
            seg = results.pop('seg').astype(int)

            # return result without transformation if tooth mask is outside image
            slices = ndimage.find_objects(seg[0, 0])
            if self.margin > 0 and (
                not slices or
                slices[0][0].start < margins[0] or
                slices[0][1].start < margins[1] or
                img.shape[0] - slices[0][0].stop < margins[0] or
                img.shape[1] - slices[0][1].stop < margins[1]
            ):
                continue
            
            # return result with transformation with tooth mask as final channel
            results['img'] = np.transpose(np.concatenate((
                np.repeat(data[0, :1], repeats=2, axis=0),
                255.0 * seg[0, :1]
            )), axes=(1, 2, 0))

            return results

        results['img'] = results['img'].astype(float)

        return results
