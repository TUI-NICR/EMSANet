# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Optional, Iterable, Tuple

from copy import deepcopy
from functools import partial
import warnings

import numpy as np
from nicr_mt_scene_analysis.data import CollateIgnoredDict
from nicr_mt_scene_analysis.data import mt_collate
from nicr_mt_scene_analysis.data import RandomSamplerSubset
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from nicr_scene_analysis_datasets.dataset_base import DatasetConfig
from nicr_scene_analysis_datasets.dataset_base import OrientationDict
from nicr_scene_analysis_datasets.dataset_base import SampleIdentifier
from nicr_scene_analysis_datasets.pytorch import DatasetType
from nicr_scene_analysis_datasets.pytorch import KNOWN_DATASETS    # noqa: F401
from nicr_scene_analysis_datasets.pytorch import KNOWN_CLASS_WEIGHTINGS    # noqa: F401
from nicr_scene_analysis_datasets.pytorch import get_dataset_class


def get_dataset(args, split):
    dataset_name = args.dataset
    dataset_path = args.dataset_path
    dataset_depth_mode = 'raw' if args.raw_depth else 'refined'
    cache_dataset = args.cache_dataset

    # determine sample keys
    sample_keys = list(args.input_modalities) + list(args.tasks)
    # add identifier for easier debugging and plotting
    sample_keys.append('identifier')
    # fix sample key for orientation
    if 'orientation' in sample_keys:
        idx = sample_keys.index('orientation')
        sample_keys[idx] = 'orientations'
    # instance task requires semantic for determing foreground
    if 'instance' in args.tasks and 'semantic' not in args.tasks:
        sample_keys.append('semantic')
    sample_keys = tuple(sample_keys)

    dataset_kwargs_dict = {
        'sunrgbd': {
            'depth_mode': dataset_depth_mode,
            'semantic_use_nyuv2_colors': True,
            'scene_use_indoor_domestic_labels': True
        },
        'nyuv2': {
            'depth_mode': dataset_depth_mode,
            'semantic_n_classes': 40,
            'scene_use_indoor_domestic_labels': True
        },
        'cityscapes': {
            'depth_mode': dataset_depth_mode,
            'semantic_n_classes': 19,
            'disparity_instead_of_depth': False
        },
        'scenenetrgbd': {
            'depth_mode': dataset_depth_mode,
            'semantic_n_classes': 13
        },
        'hypersim': {
            'depth_mode': dataset_depth_mode,
            'subsample': None,
            'scene_use_indoor_domestic_labels': True
        },
        'coco': {},
    }

    Dataset = get_dataset_class(dataset_name)
    dataset_kwargs = dataset_kwargs_dict[dataset_name]

    dataset_instance = Dataset(
        dataset_path=dataset_path,
        split=split,
        sample_keys=sample_keys,
        use_cache=cache_dataset,
        cache_disable_deepcopy=False,    # False as we modify samples inplace
        **dataset_kwargs
    )

    # TODO: can be removed from codebase later
    if 'hypersim' == args.dataset and args.hypersim_use_old_depth_stats:
        # patch dataset
        from nicr_scene_analysis_datasets import dataset_base
        dataset_instance._config = dataset_base.build_dataset_config(
            semantic_label_list=dataset_instance._config.semantic_label_list,
            scene_label_list=dataset_instance._config.scene_label_list,
            depth_stats=dataset_instance._TRAIN_SPLIT_DEPTH_STATS_V030
        )
        assert dataset_instance.depth_std == dataset_instance._TRAIN_SPLIT_DEPTH_STATS_V030.std
        assert dataset_instance.depth_mean == dataset_instance._TRAIN_SPLIT_DEPTH_STATS_V030.mean

    return dataset_instance


class DataHelper:
    def __init__(
        self,
        dataset_train: DatasetType,
        batch_size_train: int,
        datasets_valid: Iterable[DatasetType],
        batch_size_valid: Optional[int] = None,
        datasets_test: Optional[Iterable[DatasetType]] = None,
        subset_train: float = 1.0,
        subset_deterministic: bool = False,
        n_workers: int = 8,
        persistent_worker: bool = False,
    ) -> None:
        # we use a modified collate function to handle elements of different
        # spatial resolution and to ignore numpy arrays, dicts containing
        # orientations (OrientationDict), and simple tuples storing shapes
        collate_fn = partial(mt_collate,
                             type_blacklist=(np.ndarray,
                                             CollateIgnoredDict,
                                             OrientationDict,
                                             SampleIdentifier))

        # training split/set
        sampler = RandomSamplerSubset(
            data_source=dataset_train,
            subset=subset_train,
            deterministic=subset_deterministic
        )
        self._dataloader_train = DataLoader(
            dataset_train,
            batch_size=batch_size_train,
            sampler=sampler,
            drop_last=True,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=n_workers,
            persistent_workers=persistent_worker
        )

        # validation split/set
        self._dataloaders_valid = tuple(
            DataLoader(
                dataset_valid,
                batch_size=batch_size_valid or 3*batch_size_train,
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn,
                pin_memory=True,
                num_workers=n_workers,
                persistent_workers=persistent_worker
            )
            for dataset_valid in datasets_valid
        )

        # test split/set
        if datasets_test is not None:
            self._dataloaders_test = tuple(
                DataLoader(
                    dataset_test,
                    batch_size=batch_size_valid or 3*batch_size_train,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=collate_fn,
                    pin_memory=True,
                    num_workers=n_workers,
                    persistent_workers=persistent_worker
                )
                for dataset_test in datasets_test
            )
        else:
            self._dataloaders_test = None

        self._overfitting_enabled = False
        # we use the (first) valid dataset when overfitting mode gets enabled,
        # copy the dataset here to ensure that no sample was drawn before
        self._overfitting_dataset = deepcopy(self.datasets_valid[0])

    def enable_overfitting_mode(self, n_valid_batches: int) -> None:
        self._overfitting_enabled = True

        batch_size = self._dataloader_train.batch_size
        n_samples = n_valid_batches * batch_size

        dataset = self._overfitting_dataset
        camera = dataset.cameras[0]
        if len(dataset.cameras) > 1:
            warnings.warn(
                "Overfitting dataset (valid split) contains multiple cameras. "
                f"Using first camera: '{camera}' to ensure samples of same "
                "spatial resolution."
            )
        dataset.filter_camera(camera)

        if n_samples > len(dataset):
            raise ValueError(
                f"Not enough data for overfitting. Tried to draw {n_samples} "
                f"samples from {len(dataset)}. Reduce the number of batches or "
                " the batch size for overfitting!"
            )

        self._overfitting_dataloader = DataLoader(
            Subset(dataset, tuple(range(n_samples))),
            batch_size=self._dataloader_train.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=self._dataloader_train.collate_fn,
            pin_memory=True,
            num_workers=self._dataloader_train.num_workers,
            persistent_workers=self._dataloader_train.persistent_workers
        )

        print(f"Enable overfitting mode with {n_valid_batches} batches "
              f"of {batch_size} samples from validation split.")

    @property
    def dataset_config(self) -> DatasetConfig:
        # use config of train split
        return self._dataloader_train.dataset.config

    @property
    def dataset_train(self) -> DatasetType:
        return self._dataloader_train.dataset

    @property
    def datasets_valid(self) -> Tuple[DatasetType]:
        return tuple(loader.dataset for loader in self._dataloaders_valid)

    @property
    def datasets_test(self) -> Tuple[DatasetType]:
        if self._dataloaders_test is None:
            raise ValueError("No test dataset found")
        return tuple(loader.dataset for loader in self._dataloaders_test)

    def set_train_preprocessor(self, preprocessor):
        self._dataloader_train.dataset.preprocessor = preprocessor

    def set_valid_preprocessor(self, preprocessor):
        for dataset in self.datasets_valid:
            dataset.preprocessor = preprocessor

        # apply preprocessor to overfitting dataset as well
        self._overfitting_dataset.preprocessor = deepcopy(preprocessor)

    def set_test_preprocessor(self, preprocessor):
        if self._dataloaders_test is None:
            raise ValueError("No test dataset found")

        for dataset in self.datasets_test:
            dataset.preprocessor = preprocessor

    @property
    def train_dataloader(self) -> DataLoader:
        if self._overfitting_enabled:
            return self._overfitting_dataloader

        return self._dataloader_train

    @property
    def valid_dataloaders(self) -> Tuple[DataLoader]:
        if self._overfitting_enabled:
            return tuple([self._overfitting_dataloader])

        return self._dataloaders_valid

    @property
    def test_dataloaders(self) -> Tuple[DataLoader]:
        if self._overfitting_enabled:
            return tuple([self._overfitting_dataloader])

        if self._dataloaders_test is None:
            raise ValueError("No test dataset found")

        return self._dataloaders_test


def get_datahelper(args) -> DataHelper:
    # get datasets
    dataset_train = get_dataset(args, 'train')
    dataset_valid = get_dataset(args, args.validation_split)

    # create list of datasets for validation (each with only one camera ->
    # same resolution)
    dataset_valid_list = []
    for camera in dataset_valid.cameras:
        dataset_camera = deepcopy(dataset_valid).filter_camera(camera)
        dataset_valid_list.append(dataset_camera)

    if 'test' in dataset_train.SPLITS:
        # there is a separate split for testing
        # note, if there is no valid split, the test set might be equal to the
        # test set
        dataset_test = get_dataset(args, 'test')
        # create list of datasets for testing (each with only one camera ->
        # same resolution)
        dataset_test_list = []
        for camera in dataset_valid.cameras:
            dataset_camera = deepcopy(dataset_test).filter_camera(camera)
            dataset_test_list.append(dataset_camera)
    else:
        # there is no split for testing
        dataset_test_list = None

    # combine everything in a data helper
    return DataHelper(
        dataset_train=dataset_train,
        subset_train=args.subset_train,
        subset_deterministic=args.subset_deterministic,
        batch_size_train=args.batch_size,
        datasets_valid=dataset_valid_list,
        batch_size_valid=args.validation_batch_size,
        datasets_test=dataset_test_list,
        n_workers=args.n_workers,
        persistent_worker=args.cache_dataset,     # only if caching is enabled
    )
