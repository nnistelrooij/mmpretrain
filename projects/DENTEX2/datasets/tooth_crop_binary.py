from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
from tqdm import tqdm

from mmpretrain.datasets import CustomDataset
from mmpretrain.registry import DATASETS


def compute_iou(poly1, poly2, h, w):
    rles1 = maskUtils.frPyObjects(
        poly1['segmentation'], h, w,
    ) if 'size' not in poly1['segmentation'] else [poly1['segmentation']]
    rles2 = maskUtils.frPyObjects(
        poly2['segmentation'], h, w,
    ) if 'size' not in poly2['segmentation'] else [poly2['segmentation']]

    return maskUtils.iou(rles1, rles2, [0])


@DATASETS.register_module()
class ToothCropDataset(CustomDataset):

    def __init__(
        self,
        extend: float,
        pred_file: str,
        iou_thr: float=0.25,
        segm_channel: bool=True,
        mask_tooth: bool=False,
        *args, **kwargs,
    ):
        self.extend = extend
        self.pred_file = pred_file
        self.iou_thr = iou_thr
        self.segm_channel = segm_channel
        self.mask_tooth = mask_tooth

        super().__init__(*args, **kwargs)

    def crop_tooth(self, img, poly):
        height, width, _ = img.shape

        # translate bbox to xyxy
        rle = poly['segmentation']
        if 'size' not in rle:
            rle = maskUtils.frPyObjects(rle, height, width)
        bbox = maskUtils.toBbox(rle)
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
        mask = maskUtils.decode([rle]).astype(bool)

        # add extra channel of tooth segmentation
        if self.segm_channel:
            img = np.dstack((
                img[..., 0],
                img[..., 0],
                255 * mask,
            ))

        # only keep tooth in tooth crop image
        if self.mask_tooth:
            mask = mask[..., None] if mask.ndim == 2 else mask
            mask = np.repeat(mask, 3, axis=2).astype(bool)
            img[~mask] = 0

        # add mask as last channel and crop according to extended bbox
        img_crop = img[slices]

        return img_crop
    
    def save_tooth_diagnosis(self, img_path, img_crop, fdi_label, attribute):
        label_path = Path(self.img_prefix) / attribute
        label_path.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(label_path / f'{img_path.stem}_{fdi_label}.png'), img_crop)

    def crop_tooth_diagnosis(
        self,
        gt_coco: COCO,
        pred_coco: COCO,
    ):
        ann_img_stems = [Path(img_dict['file_name']).stem for _, img_dict in gt_coco.imgs.items()]
        saved_img_stems = [f.stem.split('_')[0] for f in Path(self.img_prefix).glob('*/*.png')]

        if all(stem in saved_img_stems for stem in ann_img_stems):
            return
        
        for img_id, img_dict in tqdm(gt_coco.imgs.items(), total=len(gt_coco.imgs)):
            img_path = Path(self.img_prefix) / img_dict['file_name']
            img = cv2.imread(str(img_path))

            preds = pred_coco.imgToAnns[img_id]
            if not preds:
                continue

            for poly in gt_coco.imgToAnns[img_id]:
                ious = np.zeros(len(preds))
                for i, pred_poly in enumerate(preds):
                    iou = compute_iou(
                        pred_poly, poly, img_dict['height'], img_dict['width'],
                    )
                    ious[i] = iou

                pred_poly, iou = preds[ious.argmax()], ious.max()

                if iou < self.iou_thr:
                    continue

                poly['segmentation'] = pred_poly['segmentation']
                fdi_label = pred_coco.cats[pred_poly['category_id']]['name']
                img_crop = self.crop_tooth(img, poly)

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

            gt_label = min(1, self.metainfo['attributes'].index(label))
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
        gt_coco = COCO(self.ann_file)
        pred_coco = COCO(self.pred_file)

        self.crop_tooth_diagnosis(gt_coco, pred_coco)
        data_list = self.load_annotations(gt_coco)
            
        return data_list
