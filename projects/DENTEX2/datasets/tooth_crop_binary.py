from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils

from mmpretrain.datasets import CustomDataset
from mmpretrain.registry import DATASETS


@DATASETS.register_module()
class ToothCropDataset(CustomDataset):

    def __init__(
        self,
        extend: float,
        *args, **kwargs,
    ):
        self.extend = extend

        super().__init__(*args, **kwargs)

    def crop_tooth(self, img, poly):
        height, width, _ = img.shape

        # translate bbox to xyxy
        bbox = [
            poly['bbox'][0],
            poly['bbox'][1],
            poly['bbox'][0] + poly['bbox'][2],
            poly['bbox'][1] + poly['bbox'][3],
        ]
        # extend bbox to incorporate context
        bbox = [
            max(bbox[0] - self.extend * width, 0),
            max(bbox[1] - self.extend * height, 0),
            min(width - 1, bbox[2] + self.extend * width),
            min(height - 1, bbox[3] + self.extend * height),
        ]
        # determine slices to crop imge
        slices = (
            slice(int(bbox[1]), int(bbox[3]) + 1),
            slice(int(bbox[0]), int(bbox[2]) + 1)
        )

        # determine binary mask of object in image
        rles = maskUtils.frPyObjects(poly['segmentation'], height, width)

        mask = maskUtils.decode(rles).astype(bool)
        mask = np.repeat(mask, 3, axis=-1).astype(bool)
        mask[..., :2] = 0

        # crop image and mask according to extended bbox
        img = img.copy()
        img[mask] = 255
        img_crop = img[slices]

        return img_crop
    
    def save_tooth_diagnosis(self, img_path, img_crop, fdi_label, attribute):
        label_path = Path(self.img_prefix) / attribute
        label_path.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(label_path / f'{img_path.stem}_{fdi_label}.png'), img_crop)

    def crop_tooth_diagnosis(self, coco: COCO):
        ann_img_stems = [Path(img_dict['file_name']).stem for _, img_dict in coco.imgs.items()]
        saved_img_stems = [f.stem.split('_')[0] for f in Path(self.img_prefix).glob('*/*.png')]

        if all(stem in saved_img_stems for stem in ann_img_stems):
            return
        
        for img_id, img_dict in coco.imgs.items():
            img_path = Path(self.img_prefix) / img_dict['file_name']
            img = cv2.imread(str(img_path))

            for poly in coco.imgToAnns[img_id]:
                cat_name = coco.cats[poly['category_id']]['name']
                if cat_name not in self.metainfo['classes']:
                    continue

                img_crop = self.crop_tooth(img, poly)
                fdi_label = coco.cats[poly['category_id']]['name']

                if (
                    'extra' not in poly or
                    'attributes' not in poly['extra'] or
                    len(poly['extra']['attributes']) == 0
                ):
                    self.save_tooth_diagnosis(img_path, img_crop, fdi_label, 'Control')
                    continue
                
                for attribute in poly['extra']['attributes']:
                    self.save_tooth_diagnosis(img_path, img_crop, fdi_label, attribute)

    def load_annotations(self, coco, balance=False):
        ann_img_stems = [Path(img_dict['file_name']).stem for img_dict in coco.imgs.values()]

        rng = np.random.default_rng(seed=1234)

        data_list = []
        img_paths = set()
        num_images = float('inf')
        for label in self.metainfo['attributes'][::-1]:
            label_path = Path(self.img_prefix) / label
            img_files = sorted(label_path.glob('*'))
            img_files = [img_files[idx] for idx in rng.permutation(len(img_files))]

            gt_label = self.metainfo['attributes'].index(label)
            i = 0
            for f in img_files:
                if balance and i == num_images:
                    break

                if f.stem.split('_')[0] not in ann_img_stems:
                    continue

                if str(f) in img_paths:
                    continue

                data_list.append({'img_path': str(f), 'gt_label': gt_label})
                img_paths.add(str(f))
                i += 1

            if i == len(data_list):
                num_images = len(data_list)

        return data_list

    def load_data_list(self):
        coco = COCO(self.ann_file)
        # self.crop_tooth_diagnosis(coco)
        data_list = self.load_annotations(coco)
            
        return data_list
