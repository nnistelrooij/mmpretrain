from pathlib import Path
from typing import List

import numpy as np

from mmpretrain.registry import DATASETS

from projects.DENTEX2.datasets.tooth_crop_binary import ToothCropDataset


@DATASETS.register_module()
class ToothCropMultilabelDataset(ToothCropDataset):

    def load_annotations(self, coco, max_samples=500):
        ann_img_stems = [Path(img_dict['file_name']).stem for img_dict in coco.imgs.values()]

        file_attributes = {}
        for label in self.metainfo['attributes']:
            label_path = Path(self.img_prefix) / label
            img_files = sorted(label_path.glob('*'))

            i = 0
            for f in img_files:
                if f.stem not in file_attributes and i >= max_samples:
                    continue

                if f.stem.split('_')[0] not in ann_img_stems:
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
            file_dict['gt_label'] = onehot

            data_list.append(file_dict)

        return data_list
    
    def get_cat_ids(self, idx: int) -> List[int]:
        multilabel = self.get_data_info(idx)['gt_label']

        combi_idx = sum((2 ** i) * label for i, label in enumerate(multilabel))
        
        return [combi_idx]
