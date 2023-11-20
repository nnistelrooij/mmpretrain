from collections import defaultdict
from functools import partial
import json
from multiprocessing import cpu_count, Pool
from pathlib import Path
import re
import shutil
from typing import Dict

import cv2
import numpy as np
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
from tqdm import tqdm

from mmpretrain.datasets import CustomDataset
from mmpretrain.registry import DATASETS

from projects.DENTEX2.datasets import ToothCropDataset



@DATASETS.register_module()
class ToothCropSegmentationDataset(ToothCropDataset):

    def crop_tooth(self, img, poly):
        height, width = img.shape[:2]

        # translate bbox to xyxy
        if self.iou_type == 'segm':
            rle = poly['segmentation']
            if 'size' not in rle:
                rle = maskUtils.frPyObjects(rle, height, width)
            bbox = maskUtils.toBbox(rle)
        else:
            bbox = poly['bbox']
        bbox = [
            poly['bbox'][0],
            poly['bbox'][1],
            poly['bbox'][0] + poly['bbox'][2],
            poly['bbox'][1] + poly['bbox'][3],
        ]

        # extend bbox to incorporate context
        bbox = [
            max(bbox[0] - self.extend * poly['bbox'][2], 0),
            max(bbox[1] - self.extend * poly['bbox'][3], 0),
            min(width - 1, bbox[2] + self.extend * poly['bbox'][2]),
            min(height - 1, bbox[3] + self.extend * poly['bbox'][3]),
        ]

        # determine slices to crop imge
        slices = (
            slice(int(bbox[1]), int(bbox[3]) + 1),
            slice(int(bbox[0]), int(bbox[2]) + 1)
        )

        # add extra channel of tooth segmentation
        if self.segm_channel:
            mask = maskUtils.decode([rle]).astype(bool)
            img = np.dstack((
                img[..., 0],
                img[..., 0],
                255 * mask,
            ))

        # only keep tooth in tooth crop image
        if self.mask_tooth:
            mask = maskUtils.decode([rle]).astype(bool)
            mask = mask[..., None] if mask.ndim == 2 else mask
            mask = np.repeat(mask, 3, axis=2).astype(bool)
            img[~mask] = 0

        # add mask as last channel and crop according to extended bbox
        img_crop = img[slices]
        img_irrelevant = bbox[0] == 0 or bbox[2] == width - 1

        return img_crop, img_irrelevant
    
    def save_tooth_diagnosis(self, img_path, img_crop, fdi_label, attribute) -> str:
        label_path = Path(self.img_prefix) / attribute
        label_path.mkdir(parents=True, exist_ok=True)

        out_files = sorted(label_path.glob(f'{img_path.stem}_{fdi_label}*'))
        fdi_idx = int(out_files[-1].stem[-1]) + 1 if out_files else 0
        out_file = label_path / f'{img_path.stem}_{fdi_label}-{fdi_idx}.png'

        if out_file.stem == 'Pat104_Img2_27-0':
            k = 3

        if img_crop is not None:
            cv2.imwrite(str(out_file), img_crop)

        return out_file.stem[:-2]
    
    def crop_tooth_diagnosis_fdi(
        self,
        gt_coco,
        saved_img_stems,
        img_dict,
        fdi_anns,
    ):
        img_path = Path(self.img_prefix) / img_dict['file_name']
        fdi_label = gt_coco.cats[fdi_anns[0]['category_id']]['name'][-2:]

        poly = [ann for ann in fdi_anns if gt_coco.cats[ann['category_id']]['name'].startswith('TOOTH')][0]

        img_crop, img_irrelevant = (
            self.crop_tooth(img_dict['img'], poly)
            if img_path.stem not in saved_img_stems else
            (None, None)
        )
        if img_crop is not None:
            if img_crop.shape[0] < 100 or img_crop.shape[1] < 100:
                return
            aspect_ratio = img_crop.shape[0] / img_crop.shape[1]
            if aspect_ratio < 0.5 or 2 < aspect_ratio:
                return
            if img_irrelevant:
                return
            
        attributes = [gt_coco.cats[ann['category_id']]['name'][:-3] for ann in fdi_anns]
        attributes = [attr for attr in attributes if attr in self.metainfo['classes']]
            
        if not attributes:
            label = self.metainfo['classes'][0]
            out_file = self.save_tooth_diagnosis(img_path, img_crop, fdi_label, label)
            return
        
        for i, attribute in enumerate(attributes):
            out_file = self.save_tooth_diagnosis(img_path, img_crop, fdi_label, attribute)
        
    
    def crop_tooth_diagnosis_single(
        self,
        gt_coco,
        pred_coco,
        saved_img_stems,
        img_dict,
    ):
        stem2embedding = defaultdict(lambda: defaultdict(list))
        img_path = Path(self.img_prefix) / img_dict['file_name']

        if img_path.stem not in saved_img_stems:
            img = cv2.imread(str(img_path))
            img_dict['img'] = img

        gt = gt_coco.imgToAnns[img_dict['id']]
        if not gt:
            return stem2embedding, None
        
        tooth_anns = [ann for ann in gt if gt_coco.cats[ann['category_id']]['name'].startswith('TOOTH')]
        for ann_dict in tooth_anns:
            fdi = gt_coco.cats[ann_dict['category_id']]['name'][-2:]
            fdi_anns = [ann for ann in gt if gt_coco.cats[ann['category_id']]['name'][-2:] == fdi]
            self.crop_tooth_diagnosis_fdi(
                gt_coco, saved_img_stems, img_dict, fdi_anns,
            )

    def crop_tooth_diagnosis(
        self,
        gt_coco: COCO,
        pred_coco: COCO,
    ) -> Dict:
        saved_img_stems = set([
            '_'.join(f.stem.split('_')[:-1])
            for f in Path(self.img_prefix).glob('*/*.png')
            if f.parent.name in self.metainfo['classes']
        ])

        for img_dict in tqdm(list(gt_coco.imgs.values())):
            self.crop_tooth_diagnosis_single(
                gt_coco, pred_coco, saved_img_stems, img_dict,
            )

    def load_annotations(self, coco, balance=False):
        ann_img_stems = [Path(img_dict['file_name']).stem for img_dict in coco.imgs.values()]

        if self.omit_file:
            omit_coco = COCO(self.omit_file)
            omit_img_stems = [Path(img_dict['file_name']).stem for img_dict in omit_coco.imgs.values()]
        else:
            omit_img_stems = []


        data_list = []
        img_paths = set()
        num_images = float('inf')
        for label in self.metainfo['classes']:
            label_path = Path(self.img_prefix) / label
            img_files = sorted(label_path.glob('*'))

            gt_label = self.metainfo['classes'].index(label)
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
                })
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

        self.fdi2idx = defaultdict(list)
        for i, d in enumerate(data_list):
            fdi = re.split('\.|_|-', d['img_path'])[-3]
            self.fdi2idx[fdi].append(i)
            
        return data_list
