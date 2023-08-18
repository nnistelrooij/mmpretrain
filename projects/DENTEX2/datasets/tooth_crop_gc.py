from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import re

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
import SimpleITK
from tqdm import tqdm

from mmengine.dataset.base_dataset import force_full_init
from mmengine.logging import MMLogger
from mmpretrain.datasets import CustomDataset
from mmpretrain.registry import DATASETS



@DATASETS.register_module()
class ToothCropGCDataset(CustomDataset):

    def load_images(self):
        df = pd.read_csv('/opt/algorithm/test.csv')
        df['idx'] = df['file_name'].apply(lambda fn: int(re.split('\.|_', fn)[-2]))
        df = df.sort_values(by='idx')
        df = df.reset_index(drop=True)

        mha_file = next(Path('/input/images/panoramic-dental-xrays').glob('*'))
        imgs = SimpleITK.ReadImage(mha_file)
        img_arrays = SimpleITK.GetArrayFromImage(imgs)
        img_arrays = np.transpose(img_arrays, (2, 0, 1, 3))

        images = {}
        for img_id, height, width, img in zip(
            df['id'], df['height'], df['width'], img_arrays,
        ):
            images[img_id] = img[:height, :width]

        return images
    
    def load_data_items(self, coco, inputs):
        img_id, img = inputs

        out = []
        img_dict =  {
            'ori_shape': img.shape[:2],
            'img_id': img_id,
            'img_path': Path(coco.imgs[img_id]['file_name']).stem,
            'gt_label': {task: 0 for task in self.metainfo['attributes'][1:]},
        }

        anns = coco.imgToAnns[img_id]
        crop_imgs = self.crop_teeth(img, anns)

        fdis = defaultdict(int)
        for poly, crop_img in zip(anns, crop_imgs):
            fdi = coco.cats[poly['category_id']]['name']

            data_item = img_dict.copy()
            data_item['img'] = crop_img
            data_item['img_shape'] = crop_img.shape[:2],
            data_item['img_path'] += f'_{fdi}-{fdis[fdi]}.png'
            out.append(data_item)

            fdis[fdi] += 1

        return out

    def load_data_list(self):
        MMLogger.get_current_instance().warning('loading images...')
        images = self.load_images()        
        coco = COCO(self.ann_file)

        data_list = []
        func = partial(self.load_data_items, coco)
        with Pool(8) as p:
            iterator = p.imap_unordered(func, images.items())
            for data_items in tqdm(iterator, total=len(images)):
                data_list.extend(data_items)

        MMLogger.get_current_instance().warning('Images loaded')
            
        return data_list    
        
    def crop_teeth(self, img: np.ndarray, anns):
        height, width, _ = img.shape

        # translate bbox to xyxy
        rles = []
        for poly in anns:
            rle = poly['segmentation']
            if 'size' not in rle:
                rle = maskUtils.frPyObjects(rle, height, width)
            rles.append(rle)
        
        bboxes = maskUtils.toBbox(rles)
        bboxes = np.column_stack((
            bboxes[:, 0],
            bboxes[:, 1],
            bboxes[:, 0] + bboxes[:, 2],
            bboxes[:, 1] + bboxes[:, 3],
        ))

        # extend bbox to incorporate context
        bboxes = np.column_stack((
            np.maximum(bboxes[:, 0] - 0.1 * width, 0),
            np.maximum(bboxes[:, 1] - 0.1 * height, 0),
            np.minimum(width - 1, bboxes[:, 2] + 0.1 * width),
            np.minimum(height - 1, bboxes[:, 3] + 0.1 * height),
        ))

        # determine slices to crop imge
        slices = [(
            slice(int(bbox[1]), int(bbox[3]) + 1),
            slice(int(bbox[0]), int(bbox[2]) + 1)
        ) for bbox in bboxes]

        # determine binary mask of object in image
        masks = maskUtils.decode(rles)
        masks = np.transpose(masks, (2, 0, 1))

        # add extra channel of tooth segmentation

        # add mask as last channel and crop according to extended bbox
        crop_imgs = []
        for mask, slices in zip(masks, slices):
            crop_img = np.dstack((
                img[slices][..., 0],
                img[slices][..., 0],
                255 * mask[slices],
            ))
            crop_imgs.append(crop_img.astype(np.uint8))

        return crop_imgs

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index and automatically call ``full_init`` if the
        dataset has not been fully initialized.

        Args:
            idx (int): The index of data.

        Returns:
            dict: The idx-th annotation of the dataset.
        """        
        data_info = self.data_list[idx]

        # Some codebase needs `sample_idx` of data information. Here we convert
        # the idx to a positive number and save it in data information.
        if idx >= 0:
            data_info['sample_idx'] = idx
        else:
            data_info['sample_idx'] = len(self) + idx

        return data_info
