from pathlib import Path
from typing import List

import numpy as np

from mmpretrain.registry import DATASETS

from projects.DENTEX2.datasets.tooth_crop_binary import ToothCropDataset


@DATASETS.register_module()
class ToothCropMultitaskDataset(ToothCropDataset):
    
    def __init__(
        self,
        metainfo,
        supervise_number: bool,
        *args,
        **kwargs,
    ):
        self.METAINFO['tasks'] = metainfo['attributes'][1:]
        self.supervise_number = supervise_number

        super().__init__(metainfo=metainfo, *args, **kwargs)

    def load_annotations(self, coco, max_samples=float('inf')):
        ann_img_stems = [Path(img_dict['file_name']).stem for img_dict in coco.imgs.values()]

        file_attributes = {}
        for label in self.metainfo['attributes']:
            label_path = Path(self.img_prefix) / label
            img_files = sorted(label_path.glob('*'))

            i = 0
            for f in img_files:
                if f.stem not in file_attributes and i >= max_samples:
                    continue

                if '_'.join(f.stem.split('_')[:-1]) not in ann_img_stems:
                    continue

                attribute_idx = self.metainfo['attributes'].index(label) - 1
                if f.stem in file_attributes:
                    file_attributes[f.stem]['gt_label'].append(attribute_idx)
                else:
                    file_attributes[f.stem] = {
                        'img_path': str(f),
                        'gt_label': [attribute_idx],
                    }

                i += 1

        data_list = []
        for file_dict in file_attributes.values():
            onehot = np.zeros(len(self.metainfo['attributes']) - 1, dtype=int)
            if file_dict['gt_label'][0] >= 0:
                onehot[file_dict['gt_label']] = 1
            onehot = onehot.tolist()
            
            gt_label = {task: label for task, label in zip(self.METAINFO['tasks'], onehot)}
            file_dict['gt_label'] = gt_label

            data_list.append(file_dict)

        if not self.supervise_number:
            return data_list
        
        for data_sample in data_list:
            number = Path(data_sample['img_path']).stem.split('_')[1][1]
            number_label = int(number) - 1
            data_sample['gt_label']['Number'] = number_label
        
        return data_list
    
    def get_cat_ids(self, idx: int) -> List[int]:
        labels = self.get_data_info(idx)['gt_label']
        multilabel = [label for task, label in labels.items() if task in self.metainfo['attributes']]

        combi_idx = sum((2 ** i) * label for i, label in enumerate(multilabel))
        
        return [combi_idx]
