from typing import List

import cv2
from scipy import ndimage
import numpy as np

from mmcv.transforms.utils import cache_randomness
from mmcv.transforms import BaseTransform
from mmengine.dataset import BaseDataset

from mmpretrain.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RandomToothFlip(BaseTransform):

    def __init__(
        self,
        prob: float,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.prob = prob

    @cache_randomness
    def get_indexes(self, dataset: BaseDataset) -> List[int]:
        """Call function to collect indexes.

        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.

        Returns:
            list: indexes.
        """
        fdi = np.random.choice([fdi for fdi in dataset.fdi2idx if '8' not in fdi])
        indexes = np.random.choice(dataset.fdi2idx[fdi], size=2, replace=False)
        
        return indexes.tolist()
    
    def inter_patient_copypaste(self, src_img, target_img):
        ### source ###
        intensities_src = src_img[..., 0]
        mask_src = src_img[..., 2]

        # crop #
        slice_x_src, slice_y_src = ndimage.find_objects(mask_src > 0)[0]
        tooth_crop_src = mask_src[slice_x_src,slice_y_src]
        intensities_crop_src = intensities_src[slice_x_src,slice_y_src]

        ### target ###
        intensities_target = target_img[..., 0]
        mask_target = target_img[..., 2]

        # crop #
        slice_x_target, slice_y_target = ndimage.find_objects(mask_target > 0)[0]
        tooth_crop_target = mask_target[slice_x_target,slice_y_target]
        intensities_crop_target = intensities_target[slice_x_target,slice_y_target]

        # target dims #
        len_x, len_y = slice_x_target.stop - slice_x_target.start, slice_y_target.stop - slice_y_target.start

        # postprocess #
        resized_mask = cv2.resize(tooth_crop_src, dsize=(len_y, len_x), interpolation=cv2.INTER_CUBIC) # resize the src mask to fit target dims
        resized_intensities = cv2.resize(intensities_crop_src, dsize=(len_y, len_x), interpolation=cv2.INTER_CUBIC) # resize the src intensities to fit target dims
        resized_intensities = np.where(resized_mask > 0, resized_intensities, intensities_crop_target)
        resized_intensities = np.where((tooth_crop_target > 0) & (resized_mask == 0), np.mean(intensities_target), resized_intensities) # fill overlap with mean

        intensities_output = intensities_target
        intensities_output[slice_x_target,slice_y_target] = resized_intensities


        mask_output = np.zeros(intensities_output.shape)
        mask_output[slice_x_target,slice_y_target] = resized_mask

        output = np.concatenate((
            np.repeat(intensities_output[..., None], 2, axis=-1),
            mask_output[..., None],
        ), axis=-1)

        return output
    
    def _copy_paste_tooth(
        self,
        results: dict,
    ) -> dict:
        source_results, target_results = results['mix_results']

        mixed_img = self.inter_patient_copypaste(
            source_results['img'], target_results['img'],
        )
        mixed_results = {
            'gt_label': source_results['gt_label'],
            'img': mixed_img,
            'img_shape': mixed_img.shape,
        }

        return mixed_results

    def transform(self, results: dict) -> dict:
        if np.random.rand() >= self.prob:
            return results
        
        results = self._copy_paste_tooth(results)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob})'

        return repr_str
