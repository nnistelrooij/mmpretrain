
from mmcv.transforms import BaseTransform
import numpy as np
from scipy import ndimage

from mmpretrain.registry import TRANSFORMS


@TRANSFORMS.register_module()
class MaskTooth(BaseTransform):

    def __init__(
        self,
        dilation: int=2,
    ):
        self.dilation = dilation

    def transform(self, results: dict) -> dict:
        """Transform function to randomly resized crop images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly resized cropped results, 'img_shape'
                key in result dict is updated according to crop size.
        """
        img = results['img']
        
        mask = img[..., 2] > 0
        mask = ndimage.binary_dilation(
            input=mask,
            structure=ndimage.generate_binary_structure(2, 2),
            iterations=self.dilation,
        )
        mask = np.repeat(mask[..., None], 3, axis=2)

        img = img.copy()
        img[~mask] = 0
        img = np.repeat(img[..., :1], 3, axis=2)

        results['img'] = img

        return results

    def __repr__(self):
        """Print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        return self.__class__.__name__ + f'(dilation={self.dilation})'
