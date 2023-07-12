# Copyright (c) OpenMMLab. All rights reserved.
import collections
import copy
from typing import List, Sequence, Union

from mmengine.dataset import BaseDataset, force_full_init

from mmpretrain.registry import DATASETS, TRANSFORMS


@DATASETS.register_module()
class MultiImageMixDataset:
    """A wrapper of multiple images mixed dataset.

    Suitable for training on multiple images mixed data augmentation like
    mosaic and mixup. For the augmentation pipeline of mixed image data,
    the `get_indexes` method needs to be provided to obtain the image
    indexes, and you can set `skip_flags` to change the pipeline running
    process. At the same time, we provide the `dynamic_scale` parameter
    to dynamically change the output image size.

    Args:
        dataset (:obj:`CustomDataset`): The dataset to be mixed.
        pipeline (Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        dynamic_scale (tuple[int], optional): The image scale can be changed
            dynamically. Default to None. It is deprecated.
        skip_type_keys (list[str], optional): Sequence of type string to
            be skip pipeline. Default to None.
        max_refetch (int): The maximum number of retry iterations for getting
            valid results from the pipeline. If the number of iterations is
            greater than `max_refetch`, but results is still None, then the
            iteration is terminated and raise the error. Default: 15.
    """

    def __init__(self,
                 dataset: Union[BaseDataset, dict],
                 pipeline: Sequence[str],
                 skip_type_keys: Union[Sequence[str], None] = None,
                 max_refetch: int = 15,
                 lazy_init: bool = False) -> None:
        assert isinstance(pipeline, collections.abc.Sequence)
        if skip_type_keys is not None:
            assert all([
                isinstance(skip_type_key, str)
                for skip_type_key in skip_type_keys
            ])
        self._skip_type_keys = skip_type_keys

        self.pipeline = []
        self.pipeline_types = []
        for transform in pipeline:
            if isinstance(transform, dict):
                self.pipeline_types.append(transform['type'])
                transform = TRANSFORMS.build(transform)
                self.pipeline.append(transform)
            else:
                raise TypeError('pipeline must be a dict')

        self.dataset: BaseDataset
        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
        elif isinstance(dataset, BaseDataset):
            self.dataset = dataset
        else:
            raise TypeError(
                'elements in datasets sequence should be config or '
                f'`BaseDataset` instance, but got {type(dataset)}')

        self._metainfo = self.dataset.metainfo
        if hasattr(self.dataset, 'flag'):
            self.flag = self.dataset.flag
        self.num_samples = len(self.dataset)
        self.max_refetch = max_refetch

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self) -> dict:
        """Get the meta information of the multi-image-mixed dataset.

        Returns:
            dict: The meta information of multi-image-mixed dataset.
        """
        return copy.deepcopy(self._metainfo)

    def full_init(self):
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return

        self.dataset.full_init()
        self._ori_len = len(self.dataset)
        self._fully_initialized = True

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``ConcatDataset``.

        Returns:
            dict: The idx-th annotation of the datasets.
        """
        return self.dataset.get_data_info(idx)

    @force_full_init
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        results = copy.deepcopy(self.dataset[idx])
        for (transform, transform_type) in zip(self.pipeline,
                                               self.pipeline_types):
            if self._skip_type_keys is not None and \
                    transform_type in self._skip_type_keys:
                continue

            if hasattr(transform, 'get_indexes'):
                for i in range(self.max_refetch):
                    # Make sure the results passed the loading pipeline
                    # of the original dataset is not None.
                    indexes = transform.get_indexes(self.dataset)
                    if not isinstance(indexes, collections.abc.Sequence):
                        indexes = [indexes]
                    mix_results = [
                        copy.deepcopy(self.dataset[index]) for index in indexes
                    ]
                    if None not in mix_results:
                        results['mix_results'] = mix_results
                        break
                else:
                    raise RuntimeError(
                        'The loading pipeline of the original dataset'
                        ' always return None. Please check the correctness '
                        'of the dataset and its pipeline.')

            for i in range(self.max_refetch):
                # To confirm the results passed the training pipeline
                # of the wrapper is not None.
                updated_results = transform(copy.deepcopy(results))
                if updated_results is not None:
                    results = updated_results
                    break
            else:
                raise RuntimeError(
                    'The training pipeline of the dataset wrapper'
                    ' always return None.Please check the correctness '
                    'of the dataset and its pipeline.')

            if 'mix_results' in results:
                results.pop('mix_results')

        return results

    def update_skip_type_keys(self, skip_type_keys):
        """Update skip_type_keys. It is called by an external hook.

        Args:
            skip_type_keys (list[str], optional): Sequence of type
                string to be skip pipeline.
        """
        assert all([
            isinstance(skip_type_key, str) for skip_type_key in skip_type_keys
        ])
        self._skip_type_keys = skip_type_keys

    def get_cat_ids(self, idx: int) -> List[int]:
        return self.dataset.get_cat_ids(idx)


@DATASETS.register_module()
class LabeledUnlabeledDatasets():

    def __init__(
        self,
        labeled_dataset: Union[BaseDataset, dict],
        unlabeled_dataset: Union[BaseDataset, dict],
        skip_type_keys: Union[Sequence[str], None] = None,
        max_refetch: int=15,
        lazy_init: bool=False,
    ) -> None:    
        if skip_type_keys is not None:
            assert all([
                isinstance(skip_type_key, str)
                for skip_type_key in skip_type_keys
            ])
        self._skip_type_keys = skip_type_keys

        self.labeled_dataset: BaseDataset
        if isinstance(labeled_dataset, dict):
            self.labeled_dataset = DATASETS.build(labeled_dataset)
        elif isinstance(labeled_dataset, BaseDataset):
            self.labeled_dataset = labeled_dataset
        else:
            raise TypeError(
                'elements in datasets sequence should be config or '
                f'`BaseDataset` instance, but got {type(labeled_dataset)}')

        self.unlabeled_dataset: BaseDataset
        if isinstance(unlabeled_dataset, dict):
            self.unlabeled_dataset = DATASETS.build(unlabeled_dataset)
        elif isinstance(unlabeled_dataset, BaseDataset):
            self.unlabeled_dataset = unlabeled_dataset
        else:
            raise TypeError(
                'elements in datasets sequence should be config or '
                f'`BaseDataset` instance, but got {type(unlabeled_dataset)}')

        self._metainfo = self.labeled_dataset.metainfo
        if hasattr(self.labeled_dataset, 'flag'):
            self.flag = self.labeled_dataset.flag
        self.num_samples = len(self.labeled_dataset) + len(self.unlabeled_dataset)
        self.max_refetch = max_refetch

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self) -> dict:
        """Get the meta information of the multi-image-mixed dataset.

        Returns:
            dict: The meta information of multi-image-mixed dataset.
        """
        return copy.deepcopy(self._metainfo)

    def full_init(self):
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return

        self.labeled_dataset.full_init()
        self.unlabeled_dataset.full_init()
        self._ori_len = len(self.labeled_dataset) + len(self.unlabeled_dataset)
        self._fully_initialized = True

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``ConcatDataset``.

        Returns:
            dict: The idx-th annotation of the datasets.
        """
        if idx < len(self.labeled_dataset):
            return self.labeled_dataset.get_data_info(idx)
        
        idx -= len(self.labeled_dataset)
        return self.unlabeled_dataset.get_data_info(idx)
    
    @force_full_init
    def __len__(self):
        return self.num_samples

    def update_skip_type_keys(self, skip_type_keys):
        """Update skip_type_keys. It is called by an external hook.

        Args:
            skip_type_keys (list[str], optional): Sequence of type
                string to be skip pipeline.
        """
        assert all([
            isinstance(skip_type_key, str) for skip_type_key in skip_type_keys
        ])
        self._skip_type_keys = skip_type_keys    

    def __getitem__(self, idx):
        if idx < len(self.labeled_dataset):
            results = copy.deepcopy(self.labeled_dataset[idx])
            labeled = True
        else:
            idx -= len(self.labeled_dataset)
            results = copy.deepcopy(self.unlabeled_dataset[idx])
            labeled = False

        results['data_samples'].set_field(
            labeled, 'labeled', field_type='metainfo',
        )

        return results

    def get_cat_ids(self, idx: int) -> List[int]:
        if idx < len(self.labeled_dataset):
            return self.labeled_dataset.get_cat_ids(idx)
        
        idx -= len(self.labeled_dataset)
        return self.unlabeled_dataset.get_cat_ids(idx)
