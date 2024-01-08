from collections import defaultdict
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from pycocotools.coco import COCO

from mmengine.logging import MMLogger
from mmpretrain.registry import DATASETS

from projects.DENTEX2.datasets.tooth_crop_binary import ToothCropDataset, compute_iou


@DATASETS.register_module()
class ToothCropCrowdDataset(ToothCropDataset):

    def __init__(
        self,
        ann_file: str,
        user_ids: Optional[list[int]]=None,
        aggregate: bool=True,
        *args, **kwargs,
    ):
        if user_ids is None:
            counts = self.user_id_counts(ann_file)
            user_ids = list(counts.keys())
        
        self.user_ids = sorted(user_ids)
        self.aggregate = aggregate

        super().__init__(*args, ann_file=ann_file, **kwargs)

    def user_id_counts(self, ann_file: str):
        with open(ann_file, 'r') as f:
            coco_dict = json.load(f)

        counts = defaultdict(int)
        for ann in coco_dict['annotations']:
            if not 'extra' in ann or not 'text' in ann['extra']:
                continue

            text = ann['extra']['text'].replace("'", '"')
            attributes_dict = json.loads(text)

            for ann_user_id in attributes_dict['userids']:
                counts[ann_user_id] = counts[ann_user_id] + 1

        return counts
    
    def save_tooth_diagnosis(self, img_path, img_crop, fdi_label, attribute, scores: list[float]=None) -> str:
        label_path = Path(self.img_prefix) / attribute
        label_path.mkdir(parents=True, exist_ok=True)

        out_files = sorted(label_path.glob(f'*_{img_path.stem}_{fdi_label}*'))
        fdi_idx = int(out_files[-1].stem[-1]) + 1 if out_files else 0
        scores = ','.join(map(str, scores)) if scores is not None else ','.join(['0']*len(self.user_ids))
        out_file = label_path / f'{scores}_{img_path.stem}_{fdi_label}-{fdi_idx}.png'

        if img_crop is not None:
            if not cv2.imwrite(str(out_file), img_crop):
                print('Image not written!')

        return f'{img_path.stem}_{fdi_label}'
    
    def crop_tooth_diagnosis_single(
        self,
        saved_img_stems,
        gt_cats,
        pred_cats,
        input,
        eps: float=1e-6,
    ):        
        gt_anns, pred_anns, img_dict = input

        stem2embedding = defaultdict(self.default_dict)
        img_path = Path(self.img_prefix) / img_dict['file_name']

        if img_path.stem not in saved_img_stems:
            img = cv2.imread(str(img_path))

        if not pred_anns:
            return stem2embedding, None

        ious = compute_iou(
            pred_anns, gt_anns, img_dict['height'], img_dict['width'], self.iou_type,
        )
        pred_polys = pred_anns
        for poly, ious in zip(gt_anns, ious.T):
            pred_poly, iou = pred_anns[ious.argmax()], ious.max()

            # exclude quadrant ground-truth annotations
            if len(gt_cats[poly['category_id']]['name']) == 1:
                continue
            
            # exclude non-overlapping predictions
            if iou < self.iou_thr:
                pred_polys = None
                continue

            if 'extra' in poly and 'text' in poly['extra']:
                text = poly['extra']['text'].replace("'", '"')
                poly['extra']['attributes'] = json.loads(text)

            # in case of duplicate predictions, pick class of ground truth
            if iou > 1 - eps:
                for idx in np.nonzero(ious > 1 - eps)[0]:
                    if pred_anns[idx]['category_id'] != poly['category_id']:
                        continue

                    pred_poly = pred_anns[idx]

                assert poly['category_id'] == pred_poly['category_id'], 'mismatch'

            if self.iou_type == 'segm':
                poly['segmentation'] = pred_poly['segmentation']
            fdi_label = pred_cats[pred_poly['category_id']]['name']

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
                if aspect_ratio < 0.25 or 4 < aspect_ratio:
                    continue
                if self.remove_irrelevant and img_irrelevant:
                    continue

            if (
                'extra' in pred_poly
                and 'embedding' in pred_poly['extra']
            ):
                embedding = pred_poly['extra']['embedding']
            else:
                embedding = None

            if (
                'extra' not in poly or
                'text' not in poly['extra'] or
                not any(attr in self.metainfo['classes'] for attr in poly['extra']['attributes'])
            ):
                label = self.metainfo['classes'][0]
                out_file = self.save_tooth_diagnosis(img_path, img_crop, fdi_label, label)
                stem2embedding[label][out_file].append(embedding)
                continue
            
            for attribute, value in poly['extra']['attributes'].items():
                if attribute not in self.metainfo['classes']:
                    continue

                scores = np.full(len(self.user_ids), fill_value=-1.0)
                for i, user_id in enumerate(self.user_ids):
                    if user_id not in poly['extra']['attributes']['userids']:
                        continue

                    idx = poly['extra']['attributes']['userids'].index(user_id)
                    scores[i] = value[idx]

                label = attribute if np.any(scores > 0) else self.metainfo['classes'][0]
                out_file = self.save_tooth_diagnosis(img_path, img_crop, fdi_label, label, scores)
                stem2embedding[label][out_file].append(embedding)

        return stem2embedding, pred_polys    


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
                    '_'.join(f.stem.split('_')[1:-1]) not in ann_img_stems or
                    '_'.join(f.stem.split('_')[1:-1]) in omit_img_stems
                ):
                    continue



                if str(f) in img_paths:
                    continue

                gt_percs = np.array(f.name.split('_')[0].split(',')).astype(float)
                if (gt_percs >= 0).sum() < self.min_opinions:
                    continue

                if self.aggregate:
                    gt_score = np.mean(gt_percs[gt_percs >= 0]) / 100
                    gt_scores = np.array([1 - gt_score, gt_score])
                else:
                    gt_scores = np.where(gt_percs >= 0, gt_percs / 100, -1)

                embedding = np.zeros(86, dtype=int)
                fdi = int(f.stem[-4:-2])
                embedding[fdi] = 1

                data_list.append({
                    'img_path': str(f),
                    'gt_label': gt_label,
                    'gt_score': gt_scores,
                    # 'embedding': stem2embedding[label][f.stem[6:-2]][int(f.stem[-1])],
                    'embedding': embedding,
                })
                img_paths.add(str(f))
                i += 1

            if i == len(data_list):
                num_images = len(data_list)

        is_positive = [item['gt_label'] > 0 for item in data_list]
        MMLogger.get_current_instance().info(f'Number of positive sample: {sum(is_positive)}')

        return data_list
