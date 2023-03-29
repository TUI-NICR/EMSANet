# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Optional, Tuple

from nicr_mt_scene_analysis.data.preprocessing import CloneEntries
from nicr_mt_scene_analysis.data.preprocessing import FullResCloner
from nicr_mt_scene_analysis.data.preprocessing import RandomCrop
from nicr_mt_scene_analysis.data.preprocessing import RandomHorizontalFlip
from nicr_mt_scene_analysis.data.preprocessing import InstanceClearStuffIDs
from nicr_mt_scene_analysis.data.preprocessing import InstanceTargetGenerator
from nicr_mt_scene_analysis.data.preprocessing import MultiscaleSupervisionGenerator
from nicr_mt_scene_analysis.data.preprocessing import NormalizeRGB
from nicr_mt_scene_analysis.data.preprocessing import NormalizeDepth
from nicr_mt_scene_analysis.data.preprocessing import OrientationTargetGenerator
from nicr_mt_scene_analysis.data.preprocessing import PanopticTargetGenerator
from nicr_mt_scene_analysis.data.preprocessing import RandomHSVJitter
from nicr_mt_scene_analysis.data.preprocessing import RandomResize
from nicr_mt_scene_analysis.data.preprocessing import Resize
from nicr_mt_scene_analysis.data.preprocessing import SemanticClassMapper
from nicr_mt_scene_analysis.data.preprocessing import ToTorchTensors

from nicr_scene_analysis_datasets import ScanNet

from torchvision.transforms import Compose

from .data import DatasetType
from .data import parse_datasets


def get_preprocessor(
    args,
    dataset: DatasetType,
    phase: str,
    multiscale_downscales: Optional[Tuple[int, ...]] = None,
    keep_raw_inputs=False
) -> Compose:
    assert phase in ('train', 'test')

    dataset_config = dataset.config
    sample_keys = dataset.sample_keys

    if args.visualize_validation or keep_raw_inputs:
        # clone raw inputs just to have them later for visualization
        transforms = [CloneEntries()]
    else:
        transforms = []

    # check if ScanNet benchmark mode is enabled -> handle remapping
    if 'test' == phase and args.validation_scannet_benchmark_mode:
        # enable ScanNet benchmark mode for validation ONLY, i.e., mapping
        # ignored classes to void (40 -> 20, 549 -> 200) to ignore them in
        # metrics
        assert args.scannet_semantic_n_classes in (40, 549)
        if 40 == args.scannet_semantic_n_classes:
            mapping = ScanNet.SEMANTIC_CLASSES_40_MAPPING_TO_BENCHMARK
        else:
            mapping = ScanNet.SEMANTIC_CLASSES_549_MAPPING_TO_BENCHMARK200
        classes_to_ignore = tuple(
            c_data
            for c_data, c_benchmark in mapping.items()
            if c_benchmark == 0 and c_data != 0     # ignore void
        )
        assert len(classes_to_ignore) in (40-20, 549-200)

        transforms.append(
            SemanticClassMapper(
                classes_to_map=classes_to_ignore,
                new_label=0,
            )
        )

    # check if SUNRGB-D is combined as main dataset with NYUv2, ScanNet or
    # Hypersim -> ignore last three classes (other*)
    datasets = tuple(parse_datasets(args.dataset).keys())
    if 'sunrgbd' == datasets[0]:
        if any(d in ('nyuv2', 'hypersim', 'scannet') for d in datasets[1:]):
            # map last three classes to void (ignore these classes in training/
            # validation)
            transforms.append(
                SemanticClassMapper(
                    classes_to_map=(38, 39, 40),
                    new_label=0,
                )
            )

    # instance preprocessing
    if 'instance' in sample_keys:
        # depending on the dataset and the applied division into stuff and
        # thing classes, the data may contain valid instance ids for instances
        # of stuff classes, we force id=0 (= no instance) for all stuff classes
        # including void to ensure that each stuff class is considered as a
        # single segment later
        # note that this preprocessor should be applied before resizing to
        # ensure that this requirement also applies to the full resolution
        # images that may be used for determining evaluation metrics
        is_thing = dataset_config.semantic_label_list.classes_is_thing
        transforms.append(
            InstanceClearStuffIDs(
                semantic_classes_is_thing=is_thing
            )
        )

    if 'train' == phase:
        # augmentation
        transforms.extend([
            RandomResize(
                min_scale=args.aug_scale_min,
                max_scale=args.aug_scale_max
            ),
            RandomCrop(
                crop_height=args.input_height,
                crop_width=args.input_width
            ),
            RandomHSVJitter(
                hue_jitter=10/(360/2),     # +-10 degree
                saturation_jitter=20/255,     # +- ~8%
                value_jitter=50/255,     # +- ~16%
            ),
            RandomHorizontalFlip(p=0.5)
        ])
    else:
        # panoptic targets are only required for validation
        # note: we need the panoptic targets in full resolution, so it is
        # important to have this preprocessor before the resize
        if 'semantic' in sample_keys and 'instance' in sample_keys:
            is_thing = dataset_config.semantic_label_list.classes_is_thing
            transforms.append(
                PanopticTargetGenerator(
                    semantic_classes_is_thing=is_thing,
                )
            )

        # create full-resolution copies of the relevant inputs (required for
        # resizing in inference and metrics)
        transforms.append(
            FullResCloner(
                keys_to_keep_fullres=(
                    'rgb', 'depth',    # resizing in inference
                    'semantic', 'normal', 'instance', 'panoptic'),    # eval!
                ignore_missing_keys=True    # not all keys may be available
            )
        )

        if not args.validation_full_resolution:
            # resize input images to network input resolution
            transforms.append(
                Resize(
                    height=args.validation_input_height,
                    width=args.validation_input_width,
                )
            )

    # handle mulitscale supervision
    if multiscale_downscales is not None:
        multiscale_keys = ['identifier']
        if 'semantic' in sample_keys:
            if not args.semantic_no_multiscale_supervision:
                multiscale_keys.append('semantic')

        if 'instance' in sample_keys:
            if not args.instance_no_multiscale_supervision:
                multiscale_keys.append('semantic')    # for thing vs. stuff
                multiscale_keys.append('instance')

                if 'orientations' in sample_keys:
                    multiscale_keys.append('orientations')

        if 'normal' in sample_keys:
            if not args.normal_no_multiscale_supervision:
                multiscale_keys.append('normal')

        if multiscale_keys:
            transforms.append(
                MultiscaleSupervisionGenerator(
                    downscales=multiscale_downscales,
                    keys=tuple(multiscale_keys)
                )
            )
    else:
        multiscale_downscales = ()

    # instance task
    if 'instance' in sample_keys:
        sigma = args.instance_center_sigma
        sigma_for_add_downscales = {
            downscale: (4*sigma) // downscale
            for downscale in multiscale_downscales
        }

        is_thing = dataset_config.semantic_label_list.classes_is_thing
        if args.instance_offset_encoding in ('relative', 'tanh'):
            normalized_offset = True
        else:
            normalized_offset = False
        transforms.append(
            InstanceTargetGenerator(
                sigma=sigma,
                semantic_classes_is_thing=is_thing,
                sigma_for_additional_downscales=sigma_for_add_downscales,
                normalized_offset=normalized_offset
            )
        )
    if 'orientations' in sample_keys:
        estimate_orientation = \
            dataset_config.semantic_label_list.classes_use_orientations
        transforms.append(
            OrientationTargetGenerator(
                semantic_classes_estimate_orientation=estimate_orientation
            )
        )

    # default preprocessing
    if 'rgb' in args.input_modalities or 'rgbd' in args.input_modalities:
        transforms.append(NormalizeRGB())
    if 'depth' in args.input_modalities or 'rgbd' in args.input_modalities:
        transforms.append(
            NormalizeDepth(
                depth_mean=dataset_config.depth_stats.mean,
                depth_std=dataset_config.depth_stats.std,
                raw_depth=args.raw_depth
            )
        )
    transforms.append(ToTorchTensors())

    # stack all transforms into a single preprocessor object
    preprocessor = Compose(transforms=transforms)

    if args.debug:
        print(f"Preprocessor for for phase: '{phase}':\n{preprocessor}")

    return preprocessor
