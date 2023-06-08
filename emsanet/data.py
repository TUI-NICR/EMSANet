# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Optional, Iterable, Tuple

from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict
from functools import partial
import re
import warnings

import numpy as np
from nicr_mt_scene_analysis.data import CollateIgnoredDict
from nicr_mt_scene_analysis.data import mt_collate
from nicr_mt_scene_analysis.data import RandomSamplerSubset
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from nicr_scene_analysis_datasets.dataset_base import build_dataset_config
from nicr_scene_analysis_datasets.dataset_base import DatasetConfig
from nicr_scene_analysis_datasets.dataset_base import OrientationDict
from nicr_scene_analysis_datasets.dataset_base import SampleIdentifier
from nicr_scene_analysis_datasets.dataset_base import SemanticLabel
from nicr_scene_analysis_datasets.dataset_base import SemanticLabelList
from nicr_scene_analysis_datasets.pytorch import DatasetType
from nicr_scene_analysis_datasets.pytorch import KNOWN_DATASETS    # noqa: F401
from nicr_scene_analysis_datasets.pytorch import KNOWN_CLASS_WEIGHTINGS    # noqa: F401
from nicr_scene_analysis_datasets.pytorch import ConcatDataset
from nicr_scene_analysis_datasets.pytorch import get_dataset_class
from nicr_scene_analysis_datasets.pytorch import ScanNet


class ScanNetWithOrientations(ScanNet):
    def __init__(self, **kwargs):
        # ScanNet does provide annotations for instance orientations. However,
        # we need support for orientations when combining ScanNet with other
        # datasets, e.g., NYUv2 or SUNRGB-D. This class is a workaround to
        # provide empty OrientationDicts, indicating that the instances should
        # be ignored for fitting the orientation estimation task.
        # To further use ScanNet as main dataset, this class provides a
        # function to copy the 'use_orientations' information from another
        # dataset.
        sample_keys = kwargs['sample_keys']

        # call super without orientations key
        kwargs['sample_keys'] = tuple(
            sk for sk in kwargs['sample_keys'] if sk != 'orientations'
        )
        super().__init__(**kwargs)

        # restore sample keys and re-register loaders
        self._sample_keys = sample_keys
        self.auto_register_sample_key_loaders()

        self._use_orientations_replaced = False

    def _load_orientations(self, idx):
        # we do not have instance orientations for ScanNet
        return OrientationDict({})

    @staticmethod
    def get_available_sample_keys(split: str) -> Tuple[str]:
        return ScanNet.SPLIT_SAMPLE_KEYS[split] + ('orientations',)

    def __getitem__(self, idx):
        if not self._use_orientations_replaced:
            warnings.warn(
                "You are using ScanNetWithOrientations without copying the "
                "'use_orientations' information from another dataset."
            )

        return super().__getitem__(idx)

    def copy_use_orientations_from(self, other_datataset):
        # we need another dataset to copy the 'use_orientations' information
        # from for each semantic class
        other_semantic_label_list = other_datataset.config.semantic_label_list

        # create new semantic label list
        new_semantic_label_list = SemanticLabelList()
        missing_classes = []
        for sl in self.config.semantic_label_list:
            if sl.class_name not in other_semantic_label_list:
                # we do not have this class in other_dataset
                new_semantic_label_list.add_label(sl)
                missing_classes.append(sl.class_name)
                continue

            # create new semantic label with copied 'use_orientations'
            idx = other_semantic_label_list.index(sl.class_name)
            other_sl = other_semantic_label_list[idx]
            sl_dict = asdict(sl)
            sl_dict['use_orientations'] = other_sl.use_orientations
            new_semantic_label_list.add_label(SemanticLabel(**sl_dict))

        # print warning for missing classes
        if len(missing_classes) > 0:
            warnings.warn(
                f"{self.__class__.__name__}: Could not copy 'use_orientations' "
                f"information for classes: {missing_classes} from dataset "
                f"{other_datataset.__class__.__name__}."
            )

        # replace current dataset config
        self._config = build_dataset_config(
            semantic_label_list=new_semantic_label_list,
            scene_label_list=self.config.scene_label_list,
            depth_stats=self.config.depth_stats
        )
        self._use_orientations_replaced = True


def parse_datasets(
    datasets_str: str,
    datasets_path_str: Optional[str] = None,
    datasets_split_str: Optional[str] = None
):
    misconfiguration_error = ValueError(
        "Detected dataset misconfiguration, i.e., different number of "
        f"datasets, paths or splits. Datasets: '{datasets_str}', paths: "
        f"'{datasets_path_str}', splits: '{datasets_split_str}'."
    )

    # ':' indicates joined datasets
    datasets = datasets_str.lower().split(':')
    if datasets_path_str is not None:
        dataset_paths = datasets_path_str.lower().split(':')
        if len(dataset_paths) != len(datasets):
            raise misconfiguration_error
    if datasets_split_str is not None:
        dataset_splits = datasets_split_str.lower().split(':')
        if len(dataset_splits) != len(datasets):
            raise misconfiguration_error

    dataset_dict = OrderedDict()    # we may use dict in future here (py > 3.6)
    for i, dataset in enumerate(datasets):
        # handle complex dataset format (e.g., 'sunrgbd[kv1,kv2]')
        re_res = re.findall('([a-z0-9\\_\\-]+)\\[?([a-z0-9\\_\\-]*)\\]?',
                            dataset)
        assert len(re_res) == 1 and len(re_res[0]) == 2
        # parse results (dataset_name, cameras_str)
        ds_name, ds_cameras = re_res[0]
        # split cameras
        ds_cameras = ds_cameras.split(',') if ds_cameras else None

        assert ds_name not in dataset_dict, f"Got same '{ds_name}' twice."
        dataset_dict[ds_name] = {
            'path': None if datasets_path_str is None else dataset_paths[i],
            'split': None if datasets_split_str is None else dataset_splits[i],
            'cameras': ds_cameras
        }

    return dataset_dict


def get_dataset(args, split):
    # define default kwargs dict for all datasets
    dataset_depth_mode = 'raw' if args.raw_depth else 'refined'
    default_dataset_kwargs = {
        'cityscapes': {
            'depth_mode': dataset_depth_mode,
            'semantic_n_classes': 19,
            'disparity_instead_of_depth': False
        },
        'coco': {},
        'hypersim': {
            'depth_mode': dataset_depth_mode,
            'subsample': None,
            'scene_use_indoor_domestic_labels': not args.use_original_scene_labels
        },
        'nyuv2': {
            'depth_mode': dataset_depth_mode,
            'semantic_n_classes': 40,
            'scene_use_indoor_domestic_labels': not args.use_original_scene_labels
        },
        'scannet': {
            'depth_mode': dataset_depth_mode,
            'instance_semantic_mode': 'refined',    # use refined annotations
            'scene_use_indoor_domestic_labels': not args.use_original_scene_labels,
            'semantic_n_classes': args.scannet_semantic_n_classes,
            'semantic_use_nyuv2_colors': args.scannet_semantic_n_classes in (20, 40)
        },
        'scenenetrgbd': {
            'depth_mode': dataset_depth_mode,
            'semantic_n_classes': 13
        },
        'sunrgbd': {
            'depth_mode': dataset_depth_mode,
            # note, EMSANet paper uses False for 'depth_force_mm'
            'depth_force_mm': not args.sunrgbd_depth_do_not_force_mm,
            'semantic_use_nyuv2_colors': True,
            'scene_use_indoor_domestic_labels': not args.use_original_scene_labels
        },
    }

    # prepare names, paths, and splits
    # ':' indicates joined datasets
    dataset_split = split.lower()
    n_datasets = len(parse_datasets(args.dataset))
    if 'train' == dataset_split and n_datasets > 1:
        # currently, we do not have an args for setting multiple train splits,
        # thus, we mimic the correct format
        dataset_split = ':'.join(['train'] * n_datasets)

    # parse full dataset information
    datasets = parse_datasets(
        datasets_str=args.dataset,
        datasets_path_str=args.dataset_path,
        datasets_split_str=dataset_split
    )

    # check if SUNRGB-D is combined with other datasets
    if 'sunrgbd' in datasets and len(datasets) > 1:
        # we need to force depth in mm
        warnings.warn(
            "Forcing `depth_force_mm` for SUNRGB-D, as it is combined with "
            f"other datasets. Datasets to load: {datasets}."
        )
        default_dataset_kwargs['sunrgbd']['depth_force_mm'] = True

    # determine sample keys
    sample_keys = list(args.input_modalities) + list(args.tasks)
    # add identifier for easier debugging and plotting
    sample_keys.append('identifier')
    # fix sample key for orientation
    if 'orientation' in sample_keys:
        idx = sample_keys.index('orientation')
        sample_keys[idx] = 'orientations'
    # instance task requires semantic for determining foreground
    if 'instance' in args.tasks and 'semantic' not in args.tasks:
        sample_keys.append('semantic')
    # rgbd (single encoder) modality still require rgb and depth
    if 'rgbd' in sample_keys:
        if 'rgb' not in sample_keys:
            sample_keys.append('rgb')
        if 'depth' not in sample_keys:
            sample_keys.append('depth')
        # remove rgbd key
        sample_keys.remove('rgbd')

    sample_keys = tuple(sample_keys)

    # get dataset instances
    dataset_instances = []
    for i, (dataset_name, dataset) in enumerate(datasets.items()):
        if 'none' == dataset['split']:
            # indicates that this dataset should not be loaded (e.g., for
            # training on ScanNet and SunRGB-D but validation only on SunRGB-D)
            continue

        # get dataset class
        if 'scannet' == dataset_name and 'orientations' in sample_keys:
            # we do not have orientation annotations for ScanNet, use
            # ScanNetWithOrientations as a simple workaround to mimic empty
            # OrientationDicts, however, this only makes sense if ScanNet is
            # is combined with another dataset that provides orientation
            warnings.warn(
                "Detected ScanNet dataset in dataset configuration: "
                f"{datasets} and training with orientation estimation. "
                "Switching to 'ScanNetWithOrientations' to mimic orientations."
            )
            Dataset = ScanNetWithOrientations
        else:
            Dataset = get_dataset_class(dataset_name)

        # get default kwargs for dataset
        dataset_kwargs = deepcopy(default_dataset_kwargs[dataset_name])

        # handle subsample for ScanNet
        if 'scannet' == dataset_name:
            if 'train' == dataset['split']:
                dataset_kwargs['subsample'] = args.scannet_subsample
            else:
                dataset_kwargs['subsample'] = args.validation_scannet_subsample

        # handle subsample for Hypersim
        if 'hypersim' == dataset_name:
            if 'train' == dataset['split']:
                dataset_kwargs['subsample'] = args.hypersim_subsample

        # check if all sample keys are available
        sample_keys_avail = Dataset.get_available_sample_keys(dataset['split'])
        sample_keys_missing = set(sample_keys) - set(sample_keys_avail)

        if sample_keys_missing:
            # this indicates a common problem, however, it also happens for
            # inference ScanNet on test split
            warnings.warn(
                f"Sample keys '{sample_keys_missing}' are not available for "
                f"dataset '{dataset_name}' and split '{dataset['split']}'. "
                "Removing them from sample keys."
            )
            sample_keys = tuple(set(sample_keys) - sample_keys_missing)

        # instantiate dataset object
        dataset_instance = Dataset(
            dataset_path=dataset['path'],
            split=dataset['split'],
            sample_keys=sample_keys,
            use_cache=args.cache_dataset,
            cache_disable_deepcopy=False,    # False as we modify samples inplace
            cameras=dataset['cameras'],
            **dataset_kwargs
        )

        # TODO: can be removed from codebase later
        if 'hypersim' == dataset_name and args.hypersim_use_old_depth_stats:
            # patch dataset
            from nicr_scene_analysis_datasets import dataset_base
            dataset_instance._config = dataset_base.build_dataset_config(
                semantic_label_list=dataset_instance._config.semantic_label_list,
                scene_label_list=dataset_instance._config.scene_label_list,
                depth_stats=dataset_instance._TRAIN_SPLIT_DEPTH_STATS_V030
            )
            assert dataset_instance.depth_std == dataset_instance._TRAIN_SPLIT_DEPTH_STATS_V030.std
            assert dataset_instance.depth_mean == dataset_instance._TRAIN_SPLIT_DEPTH_STATS_V030.mean

        dataset_instances.append(dataset_instance)

    if 1 == len(dataset_instances):
        # single dataset
        return dataset_instances[0]

    if isinstance(dataset_instances[0], ScanNetWithOrientations):
        # we switched from ScanNet to ScanNetWithOrientations as it is combined
        # with other datasets, however, we do not have valid 'use_orientations'
        # information for ScanNet, so we copy it from the next dataset
        dataset_instances[0].copy_use_orientations_from(dataset_instances[1])

    # concatenated datasets
    return ConcatDataset(dataset_instances[0], *dataset_instances[1:])


class DataHelper:
    def __init__(
        self,
        dataset_train: DatasetType,
        batch_size_train: int,
        datasets_valid: Iterable[DatasetType],
        batch_size_valid: Optional[int] = None,
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

        print("Enable overfitting mode (same data for training and validation) "
              f"with {n_valid_batches} batches (each with {batch_size} "
              "samples) from validation split.")

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

    def set_train_preprocessor(self, preprocessor):
        self._dataloader_train.dataset.preprocessor = preprocessor

    def set_valid_preprocessor(self, preprocessor):
        for dataset in self.datasets_valid:
            dataset.preprocessor = preprocessor

        # apply preprocessor to overfitting dataset as well
        self._overfitting_dataset.preprocessor = deepcopy(preprocessor)

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

    # combine everything in a data helper
    return DataHelper(
        dataset_train=dataset_train,
        subset_train=args.subset_train,
        subset_deterministic=args.subset_deterministic,
        batch_size_train=args.batch_size,
        datasets_valid=dataset_valid_list,
        batch_size_valid=args.validation_batch_size,
        n_workers=args.n_workers,
        persistent_worker=args.cache_dataset,     # only if caching is enabled
    )
