from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import re
from typing import Dict, Tuple

import cv2
import numpy as np
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
from tqdm import tqdm

from mmpretrain.datasets import CustomDataset
from mmpretrain.registry import DATASETS

def convert(poly, h, w):
    return (
        maskUtils.frPyObjects(
            poly['segmentation'], h, w,
        )[0] if 'size' not in poly['segmentation'] else poly['segmentation']
    )



def compute_iou(polys1, polys2, h, w):
    d = [convert(poly, h, w) for poly in polys1]
    g = [convert(poly, h, w) for poly in polys2]

    return maskUtils.iou(d, g, [0]*len(g))


@DATASETS.register_module()
class ToothCropDataset(CustomDataset):

    def __init__(
        self,
        extend: float,
        pred_file: str,
        omit_file: str='',
        iou_thr: float=0.25,
        segm_channel: bool=True,
        mask_tooth: bool=False,
        *args, **kwargs,
    ):
        self.extend = extend
        self.pred_file = pred_file
        self.omit_file = omit_file
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
    
    def save_tooth_diagnosis(self, img_path, img_crop, fdi_label, attribute) -> Path:
        label_path = Path(self.img_prefix) / attribute
        label_path.mkdir(parents=True, exist_ok=True)

        out_file = label_path / f'{img_path.stem}_{fdi_label}.png'
        if img_crop is not None:
            cv2.imwrite(str(out_file), img_crop)

        return out_file
    
    def crop_tooth_diagnosis_single(
        self,
        gt_coco,
        pred_coco,
        saved_img_stems,
        img_dict,
        score_thrs={
            'Caries': (0.1, 0.8),
            'Deep Caries': (0.1, 0.8),
            'Periapical Lesion': (0.1, 0.9),
            'Impacted': (0.1, 0.8),
        }
    ):
        stem2logits = defaultdict(int)
        img_path = Path(self.img_prefix) / img_dict['file_name']

        if img_path.stem not in saved_img_stems:
            img = cv2.imread(str(img_path))

        gt = gt_coco.imgToAnns[img_dict['id']]
        preds = pred_coco.imgToAnns[img_dict['id']]
        if not preds:
            return

        ious = compute_iou(preds, gt, img_dict['height'], img_dict['width'])
        for poly, ious in zip(gt, ious.T):
            pred_poly, iou = preds[ious.argmax()], ious.max()

            if iou < self.iou_thr:
                continue

            poly['segmentation'] = pred_poly['segmentation']
            fdi_label = pred_coco.cats[pred_poly['category_id']]['name']

            img_crop = (
                self.crop_tooth(img, poly)
                if img_path.stem not in saved_img_stems else
                None
            )

            if (
                'extra' in pred_poly
                and 'logits' in pred_poly['extra']
            ):
                stem2logits[f'{img_path.stem}_{fdi_label}'] = pred_poly['extra']['logits']
            else:
                stem2logits[f'{img_path.stem}_{fdi_label}'] = 0
                

            if (
                'extra' not in poly or
                'attributes' not in poly['extra'] or
                len(poly['extra']['attributes']) == 0
            ):
                self.save_tooth_diagnosis(img_path, img_crop, fdi_label, 'Control')
                continue
            
            for i, attribute in enumerate(poly['extra']['attributes']):
                if 'scores' in poly['extra'] and poly['extra']['scores'][i] < score_thrs[attribute][1]:
                    continue

                self.save_tooth_diagnosis(img_path, img_crop, fdi_label, attribute)

            if (
                'scores' in poly['extra'] and
                all(score < score_thrs[attr][0] for attr, score in zip(*poly['extra'].values()))
            ):
                self.save_tooth_diagnosis(img_path, img_crop, fdi_label, 'Control')

        return stem2logits

    def crop_tooth_diagnosis(
        self,
        gt_coco: COCO,
        pred_coco: COCO,
    ) -> Dict:
        saved_img_stems = set([
            '_'.join(f.stem.split('_')[:-1])
            for f in Path(self.img_prefix).glob('*/*.png')
        ])
        
        stem2logits = defaultdict(int)
        p = Pool(8)
        iterator = p.imap_unordered(
            func=partial(self.crop_tooth_diagnosis_single, gt_coco, pred_coco, saved_img_stems),
            iterable=gt_coco.dataset['images'],
        )
        for logits_dict in tqdm(iterator, total=len(gt_coco.imgs)):
            stem2logits.update(logits_dict)

        return stem2logits

    def load_annotations(self, stem2logits, coco, balance=False):
        ann_img_stems = [Path(img_dict['file_name']).stem for img_dict in coco.imgs.values()]

        if self.omit_file:
            omit_coco = COCO(self.omit_file)
            omit_img_stems = [Path(img_dict['file_name']).stem for img_dict in omit_coco.imgs.values()]
        else:
            omit_img_stems = []


        data_list = []
        img_paths = set()
        num_images = float('inf')
        for label in self.metainfo['attributes'][::-1]:
            label_path = Path(self.img_prefix) / label
            img_files = sorted(label_path.glob('*'))

            gt_label = min(1, self.metainfo['attributes'].index(label))
            i = 0
            for f in img_files:
                if balance and i == num_images:
                    break

                if (
                    '_'.join(f.stem.split('_')[:-1]) not in ann_img_stems or
                    '_'.join(f.stem.split('_')[:-1]) in omit_img_stems
                ):
                    continue



                if str(f) in img_paths:
                    continue

                data_list.append({
                    'img_path': str(f),
                    'gt_label': gt_label,
                    'logits': stem2logits[f.stem],
                })
                img_paths.add(str(f))
                i += 1

            if i == len(data_list):
                num_images = len(data_list)

        return data_list

    def load_data_list(self):
        gt_coco = COCO(self.ann_file)
        pred_coco = COCO(self.pred_file)

        # stem2logits = self.crop_tooth_diagnosis(gt_coco, pred_coco)
        stem2logits = defaultdict(int)
        data_list = self.load_annotations(stem2logits, gt_coco)

        self.fdi2idx = defaultdict(list)
        for i, d in enumerate(data_list):
            fdi = re.split('\.|_', d['img_path'])[-2]
            self.fdi2idx[fdi].append(i)
            
        return data_list
