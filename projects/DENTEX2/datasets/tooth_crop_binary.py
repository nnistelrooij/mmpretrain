from collections import defaultdict
from functools import partial
import json
from multiprocessing import cpu_count, Pool
from pathlib import Path
import re
import shutil
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



def compute_iou(polys1, polys2, h, w, iou_type):
    if iou_type == 'segm':
        d = [convert(poly, h, w) for poly in polys1]
        g = [convert(poly, h, w) for poly in polys2]
    elif iou_type == 'bbox':
        d = np.stack([poly['bbox'] for poly in polys1])
        g = np.stack([poly['bbox'] for poly in polys2])

    ious = maskUtils.iou(d, g, [0]*len(g))

    if isinstance(ious, list):
        ious = np.zeros((0, 0))

    return ious


@DATASETS.register_module()
class ToothCropDataset(CustomDataset):

    def __init__(
        self,
        extend: float,
        pred_file: str,
        omit_file: str='',
        iou_thr: float=0.25,
        iou_type: str='segm',
        segm_channel: bool=True,
        mask_tooth: bool=False,
        num_workers: int=0,
        min_shape: Tuple[int, int]=(100, 100),
        *args, **kwargs,
    ):
        assert iou_type == 'segm' or not segm_channel
        assert iou_type == 'segm' or not mask_tooth

        self.extend = extend
        self.pred_file = pred_file
        self.omit_file = omit_file
        self.iou_thr = iou_thr
        self.iou_type = iou_type
        self.segm_channel = segm_channel
        self.mask_tooth = mask_tooth
        self.num_workers = num_workers
        self.min_shape = min_shape

        super().__init__(*args, **kwargs)

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

        if img_crop is not None:
            cv2.imwrite(str(out_file), img_crop)

        return out_file.stem[:-2]
    
    def crop_tooth_diagnosis_single(
        self,
        gt_coco,
        pred_coco,
        saved_img_stems,
        img_dict,
        score_thrs={
            'Caries': (0.8, 0.8),
            'Deep Caries': (0.8, 0.8),
            'Periapical Lesion': (0.9, 0.9),
            'Impacted': (0.8, 0.8),
        },
        eps: float=1e-6,
    ):
        stem2embedding = defaultdict(lambda: defaultdict(list))
        img_path = Path(self.img_prefix) / img_dict['file_name']

        if img_path.stem not in saved_img_stems:
            img = cv2.imread(str(img_path))

        gt = gt_coco.imgToAnns[img_dict['id']]
        preds = pred_coco.imgToAnns[img_dict['id']]
        if not preds:
            return stem2embedding, None

        ious = compute_iou(
            preds, gt, img_dict['height'], img_dict['width'], self.iou_type,
        )
        pred_polys = preds
        for poly, ious in zip(gt, ious.T):
            pred_poly, iou = preds[ious.argmax()], ious.max()

            # exclude quadrant ground-truth annotations
            if len(gt_coco.cats[poly['category_id']]['name']) == 1:
                continue
            
            # exclude non-overlapping predictions
            if iou < self.iou_thr:
                pred_polys = None
                continue

            if 'extra' in poly and 'attributes' in poly['extra']:
                pred_poly['extra']['attributes'] = poly['extra']['attributes']

            # in case of duplicate predictions, pick class of ground truth
            if iou > 1 - eps:
                for idx in np.nonzero(ious > 1 - eps)[0]:
                    if preds[idx]['category_id'] != poly['category_id']:
                        continue

                    pred_poly = preds[idx]

                assert poly['category_id'] == pred_poly['category_id'], 'mismatch'

            if self.iou_type == 'segm':
                poly['segmentation'] = pred_poly['segmentation']
            fdi_label = pred_coco.cats[pred_poly['category_id']]['name']

            img_crop, img_irrelevant = (
                self.crop_tooth(img, poly)
                if img_path.stem not in saved_img_stems else
                (None, None)
            )
            if img_crop is not None:
                if (
                    img_crop.shape[0] < self.min_shape[0]
                    or img_crop.shape[1] < self.min_shape[1]
                ):
                    continue
                aspect_ratio = img_crop.shape[0] / img_crop.shape[1]
                if aspect_ratio < 0.5 or 2 < aspect_ratio:
                    continue
                if img_irrelevant:
                    continue

            if (
                'extra' in pred_poly
                and 'embedding' in pred_poly['extra']
            ):
                embedding = pred_poly['extra']['embedding']
            else:
                None                

            if (
                'extra' not in poly or
                'attributes' not in poly['extra'] or
                not any(attr in self.metainfo['classes'] for attr in poly['extra']['attributes'])
            ):
                label = self.metainfo['classes'][0]
                out_file = self.save_tooth_diagnosis(img_path, img_crop, fdi_label, label)
                stem2embedding[label][out_file].append(embedding)
                continue
            
            for i, attribute in enumerate(poly['extra']['attributes']):
                if attribute not in self.metainfo['classes']:
                    continue

                if 'scores' in poly['extra'] and poly['extra']['scores'][i] < score_thrs[attribute][1]:
                    continue

                out_file = self.save_tooth_diagnosis(img_path, img_crop, fdi_label, attribute)
                stem2embedding[attribute][out_file].append(embedding)

            if (
                'scores' in poly['extra'] and
                all(score < score_thrs[attr][0] for attr, score in zip(*poly['extra'].values()))
            ):
                label = self.metainfo['classes'][0]
                out_file = self.save_tooth_diagnosis(img_path, img_crop, fdi_label, label)
                stem2embedding[label][out_file].append(embedding)

        return stem2embedding, pred_polys

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

        stem2embedding = defaultdict(lambda: defaultdict(list))
        pred_img_anns = []
        if not self.num_workers:
            for img_dict in tqdm(list(gt_coco.imgs.values())):
                embedding_dict, pred_anns = self.crop_tooth_diagnosis_single(
                    gt_coco, pred_coco, saved_img_stems, img_dict,
                )
                {stem2embedding[k].update(embedding_dict[k]) for k in embedding_dict}
                pred_img_anns.append(pred_anns)
        else:        
            with Pool(self.num_workers) as p:
                iterator = p.imap_unordered(
                    func=partial(self.crop_tooth_diagnosis_single, gt_coco, pred_coco, saved_img_stems),
                    iterable=gt_coco.imgs.values(),
                )
                for embedding_dict, pred_anns in tqdm(iterator, total=len(gt_coco.imgs)):
                    {stem2embedding[k].update(embedding_dict[k]) for k in embedding_dict}
                    pred_img_anns.append(pred_anns)

        coco_dict = {
            'images': [img for img, img_anns in zip(gt_coco.imgs.values(), pred_img_anns) if img_anns is not None],
            'categories': list(pred_coco.cats.values()),
            **(
                {'tag_categories': pred_coco.dataset['tag_categories']}
                if 'tag_categories' in pred_coco.dataset else
                {}
            ),
        }
        pred_img_anns = [img_anns for img_anns in pred_img_anns if img_anns is not None]
        coco_dict['annotations'] = [ann for img_anns in pred_img_anns for ann in img_anns]
        
        with open('pred_odo_diagnoses.json', 'w') as f:
            json.dump(coco_dict, f, indent=2)

        return stem2embedding

    def load_annotations(self, stem2embedding, coco, balance=False):
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
                    'embedding': stem2embedding[label][f.stem[:-2]][int(f.stem[-1])],
                })
                img_paths.add(str(f))
                i += 1

            if i == len(data_list):
                num_images = len(data_list)

        return data_list

    def load_data_list(self):
        gt_coco = COCO(self.ann_file)
        pred_coco = COCO(self.pred_file)

        stem2embedding = self.crop_tooth_diagnosis(gt_coco, pred_coco)
        data_list = self.load_annotations(stem2embedding, gt_coco)

        self.fdi2idx = defaultdict(list)
        for i, d in enumerate(data_list):
            fdi = re.split('\.|_|-', d['img_path'])[-3]
            self.fdi2idx[fdi].append(i)
            
        return data_list
