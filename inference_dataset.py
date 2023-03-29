# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
from copy import deepcopy
from datetime import datetime
from functools import partial
import getpass
import json
import os
from pprint import pprint
import sys
from time import time

import warnings

import cv2
import numpy as np
import torch
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
from torch.utils.data import DataLoader
from tqdm import tqdm

from nicr_mt_scene_analysis.data import move_batch_to_device
from nicr_mt_scene_analysis.data import mt_collate
from nicr_mt_scene_analysis.data import CollateIgnoredDict
from nicr_mt_scene_analysis.data.preprocessing.resize import get_fullres
from nicr_mt_scene_analysis.data.preprocessing.resize import get_fullres_key

from nicr_scene_analysis_datasets import ScanNet
from nicr_scene_analysis_datasets.dataset_base import OrientationDict
from nicr_scene_analysis_datasets.dataset_base import SampleIdentifier

from emsanet.args import ArgParserEMSANet
from emsanet.data import get_dataset
from emsanet.model import EMSANet
from emsanet.preprocessing import get_preprocessor
from emsanet.weights import load_weights


_SCORE_MAX = 0.999


def _get_args():
    parser = ArgParserEMSANet()

    # add additional arguments
    group = parser.add_argument_group('Inference')
    group.add_argument(
        '--inference-split',
        type=str,
        default='test',
        help="Dataset split to load."
    )
    group.add_argument(
        '--inference-scannet-subsample',
        type=int,
        default=100,
        choices=(5, 10, 50, 100, 200, 500),    # 5 only for mapping inference
        help="Subsample to use for ScanNet dataset."
    )
    group.add_argument(    # useful for appm context module
        '--inference-input-height',
        type=int,
        default=480,
        dest='validation_input_height',    # used in test phase
        help="Network input height for predicting on inference data."
    )
    group.add_argument(    # useful for appm context module
        '--inference-input-width',
        type=int,
        default=640,
        dest='validation_input_width',    # used in test phase
        help="Network input width for predicting on inference data."
    )
    group.add_argument(
        '--inference-batch-size',
        type=int,
        default=8,
        help="Batch size to use for inference."
    )
    group.add_argument(
        '--inference-output-path',
        type=str,
        default=None,
        help="Path where to write inference outputs to."
    )
    group.add_argument(
        '--inference-output-format',
        type=str,
        nargs='+',
        default='scannet-semantic',
        choices=('scannet-semantic', 'scannet-instance', 'scannet-panoptic',
                 'mapping'),
        help="Output format(s) for inference."
    )
    group.add_argument(
        '--inference-output-write-ground-truth',
        action='store_true',
        default=False,
        help="For output format 'scannet-*', write ground-truth data."
    )
    group.add_argument(
        '--inference-output-ground-truth-max-depth',
        type=float,
        default=None,
        help="Mask all ground-truth annotations with depth larger then this "
             "value (in m) to void. By default, no masking is performed."
    )
    group.add_argument(
        '--inference-output-semantic-instance-shift',
        type=int,
        default=1000,
        choices=(1000, (1 << 16)),
        help="Shift to apply for writing ground-truth annotations for output "
             "format 'scannet-instance'. ScanNet benchmark by default uses "
             "1000 and encodes ground-truth instances as sem*1000+inst. "
             "However, for Hypersim, 1000 is too small, thus, we use "
             "(1<<16=2^16) instead. Note that shifting 16 bits also requires "
             "changing the output format as annotations cannot be stored in a "
             "png16 anymore. Similar to the panoptic encoding, we use a png8 "
             "with three channels instead: R: semantic class (uint8), G+B: "
             "instance id (uint16)."
    )
    group.add_argument(
        '--overwrite',
        action='store_true',
        default=False,
        help="Force overwriting of existing output files."
    )

    args = parser.parse_args()

    return args


def _semantic_and_instance_to_panoptic_bgr(semantic, instance):
    assert semantic.max() <= np.iinfo('uint8').max
    semantic_uint8 = semantic.astype('uint8')

    assert instance.shape == semantic.shape
    assert instance.max() <= np.iinfo('uint16').max
    instance_uint16 = instance.astype('uint16')

    r = semantic_uint8                              # semantic class
    g = (instance_uint16 >> 8).astype('uint8')      # upper 8bit of instance id
    b = (instance_uint16 & 0xFF).astype('uint8')    # lower 8bit of instance id

    # BGR for opencv
    panoptic_img = np.stack([b, g, r], axis=2)

    return panoptic_img


def write_scannet_panoptic_output(
    batch,
    prediction,
    output_path,
    max_instances_per_category,
    identifier_to_filename_mapper,
    max_depth=None,
    semantic_class_mapper=lambda x: x,
    write_gt=False
):
    # For evaluating the panoptic segmentation, we need to save the
    # image in the following format:
    # R: semantic class (uint8), G+B: instance id (uint16)
    # We save the image in the following format:
    # unzip_root/
    # |-- scene0707_00_000000.png
    # |-- scene0707_00_000200.png
    # |-- scene0707_00_000400.png
    #     ⋮

    # Note that, for Hypersim, semantic and panoptic_semantic (i.e. semantic
    # after merging semantic and instance) slightly differ for few images.
    # This is because there are some pixels that belong to a thing class but
    # are not assigned to any instance (instance=0), e.g., in scene ai_052_001,
    # a lamp is labeled as lamp but is not annotated as instance. Panoptic
    # merging assigns void for those pixels. There is no workaround for this
    # issue. Affected scenes: valid: ai_023_003, ai_041_003, ai_052_001,
    # ai_052_003 -> 1576566 pixels (0.03%); test: ai_005_001, ai_008_005,
    # ai_008_005, ai_022_001 -> 801359 pixels (0.01%).
    # Computing mIoU in [0, 1] to semantic / panoptic_semantic as ground truth
    # changes the result by ~0.0001-0.0002 - so it is not a big issue and
    # negligible.

    # ground-truth panoptic
    # read semantic and instance and combine them to panoptic
    if write_gt and get_fullres_key('panoptic') in batch:
        path = os.path.join(output_path, 'gt_path')
        os.makedirs(path, exist_ok=True)

        gt_panoptics = get_fullres(batch, 'panoptic').cpu().numpy()
        for i, (gt_panoptic) in enumerate(gt_panoptics):
            # extract semantic and instance from merged panoptic
            gt_semantic = gt_panoptic // max_instances_per_category
            gt_instance = gt_panoptic % max_instances_per_category

            # apply opt. class mapping
            gt_semantic = semantic_class_mapper(gt_semantic)

            # mask out all pixels with depth larger then max_depth
            if max_depth is not None:
                depth = batch['_no_preprocessing']['depth'][i]
                depth_mask = depth > max_depth
                gt_semantic[depth_mask] = 0
                gt_instance[depth_mask] = 0

            cv2.imwrite(
                os.path.join(
                    path,
                    identifier_to_filename_mapper(batch['identifier'][i])
                ),
                _semantic_and_instance_to_panoptic_bgr(gt_semantic, gt_instance)
            )

    # predicted panoptic
    path = os.path.join(output_path, 'pred_path')
    os.makedirs(path, exist_ok=True)
    panoptic_segmentation_semantic = get_fullres(prediction, 'panoptic_segmentation_deeplab_semantic_idx').cpu().numpy()
    panoptic_segmentation_semantic = semantic_class_mapper(panoptic_segmentation_semantic)    # map classes
    panoptic_segmentation_instance = get_fullres(prediction, 'panoptic_segmentation_deeplab_instance_idx').cpu().numpy()

    for b_idx in range(panoptic_segmentation_semantic.shape[0]):
        cv2.imwrite(
            os.path.join(
                path,
                identifier_to_filename_mapper(batch['identifier'][b_idx])
            ),
            _semantic_and_instance_to_panoptic_bgr(
                panoptic_segmentation_semantic[b_idx],
                panoptic_segmentation_instance[b_idx]
            )
        )


def write_scannet_semantic_output(
    batch,
    prediction,
    output_path,
    identifier_to_filename_mapper,
    max_depth=None,
    semantic_class_mapper=lambda x: x,
    write_gt=False
):
    # Scannet benchmark format for semantic segmentation
    # see: https://kaldir.vc.in.tum.de/scannet_benchmark/documentation#format-label2d
    # see: https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/2d_evaluation/evalPixelLevelSemanticLabeling.py
    # format:
    # unzip_root/
    # |-- scene0707_00_000000.png
    # |-- scene0707_00_000200.png
    # |-- scene0707_00_000400.png
    #     ⋮

    # ground-truth semantic
    if write_gt and get_fullres_key('semantic') in batch:
        path = os.path.join(output_path, 'gt_path')
        os.makedirs(path, exist_ok=True)

        for i, gt in enumerate(get_fullres(batch, 'semantic').cpu().numpy()):
            gt_semantic = semantic_class_mapper(gt)  # gt has void class

            # mask out all pixels with depth larger then max_depth
            if max_depth is not None:
                depth = batch['_no_preprocessing']['depth'][i]
                depth_mask = depth > max_depth
                gt_semantic[depth_mask] = 0

            cv2.imwrite(
                os.path.join(
                    path,
                    identifier_to_filename_mapper(batch['identifier'][i])
                ),
                gt_semantic
            )

    # semantic prediction
    path = os.path.join(output_path, 'pred_path_semantic')
    os.makedirs(path, exist_ok=True)
    pred_semantic = get_fullres(prediction, 'semantic_segmentation_idx')
    pred_semantic = pred_semantic.to(torch.uint8).cpu().numpy()
    for i, pred in enumerate(pred_semantic):
        cv2.imwrite(
            os.path.join(
                path,
                identifier_to_filename_mapper(batch['identifier'][i])
            ),
            semantic_class_mapper(pred + 1)    # add 0 as void class
        )

    # panoptic semantic prediction
    path = os.path.join(output_path, 'pred_path_panoptic_semantic')
    os.makedirs(path, exist_ok=True)
    pred_semantic = get_fullres(prediction, 'panoptic_segmentation_deeplab_semantic_idx')
    pred_semantic = pred_semantic.to(torch.uint8).cpu().numpy()
    for i, pred in enumerate(pred_semantic):
        cv2.imwrite(
            os.path.join(
                path,
                identifier_to_filename_mapper(batch['identifier'][i])
            ),
            semantic_class_mapper(pred)    # already has void class
        )


def write_scannet_instance_output(
    batch,
    prediction,
    output_path,
    identifier_to_filename_mapper,
    shift=1000,
    max_depth=None,
    semantic_class_mapper=lambda x: x,
    write_gt=False
):
    # Scannet benchmark format for instance segmentation
    # see: https://kaldir.vc.in.tum.de/scannet_benchmark/documentation#format-instance2d
    # see: https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/2d_evaluation/evalInstanceLevelSemanticLabeling.py
    # prediction format:
    # root/
    # |-- scene0707_00_000000.txt
    # |-- scene0707_00_000200.txt
    # |-- scene0707_00_000400.txt
    #     ⋮
    # |-- predicted_masks/
    #     |-- scene0707_00_000000_000.png
    #     |-- scene0707_00_000000_001.png
    #         ⋮
    # with scene0707_00_000000.txt containing:
    # predicted_masks/scene0707_00_000000_000.png 33 0.7234
    # predicted_masks/scene0707_00_000000_001.png 5 0.9038

    # ground-truth semantic+instance
    # see: https://github.com/ScanNet/ScanNet/blob/3e5726500896748521a6ceb81271b0f5b2c0e7d2/BenchmarkScripts/2d_helpers/convert_scannet_instance_image.py

    # ScanNet benchmark by default uses 1000 and encodes ground-truth instances
    # as sem*1000+inst. However, for Hypersim, 1000 is too small, thus, we
    # use (1<<16=2^16) instead. Note that shifting 16 bits also requires
    # changing the output format as annotations cannot be stored in a png16
    # anymore. Similar to the panoptic encoding, we use a png8 with three
    # channels instead: R: semantic class (uint8), G+B: instance id (uint16).
    assert shift in (1000, (1 << 16))

    if write_gt and all(get_fullres_key(k) in batch for k in ('semantic',
                                                              'instance')):
        path = os.path.join(output_path, 'gt_path')
        os.makedirs(path, exist_ok=True)
        gt_semantic = get_fullres(batch, 'semantic').cpu().numpy()
        gt_instance = get_fullres(batch, 'instance').cpu().numpy()

        if 1000 == shift:
            # scannet default shift

            # apply opt. class mapping
            gt_semantic_instance = semantic_class_mapper(gt_semantic)

            # create combined label as label * 1000 + instance_id
            gt_semantic_instance = gt_semantic_instance.astype('uint16') * 1000
            gt_semantic_instance += gt_instance.astype('uint16')

            for i, gt in enumerate(gt_semantic_instance):
                # mask out all pixels with depth larger then max_depth
                if max_depth is not None:
                    depth = batch['_no_preprocessing']['depth'][i]
                    depth_mask = depth > max_depth
                    gt[depth_mask] = 0

                cv2.imwrite(
                    os.path.join(
                        path,
                        identifier_to_filename_mapper(batch['identifier'][i])
                    ),
                    gt
                )
        else:
            # scannet shift by 2^16 (three channel encoding)

            for i, (gt_sem, gt_ins) in enumerate(zip(gt_semantic, gt_instance)):
                # apply opt. class mapping
                gt_sem = semantic_class_mapper(gt_sem)

                # mask out all pixels with depth larger then max_depth
                if max_depth is not None:
                    depth = batch['_no_preprocessing']['depth'][i]
                    depth_mask = depth > max_depth
                    gt_sem[depth_mask] = 0
                    gt_ins[depth_mask] = 0

                cv2.imwrite(
                    os.path.join(
                        path,
                        identifier_to_filename_mapper(batch['identifier'][i])
                    ),
                    _semantic_and_instance_to_panoptic_bgr(gt_sem, gt_ins)
                )

    # TODO: instance prediction with gt mask (where to get the semantic from?)
    # prediction['instance_segmentation_gt_foreground_fullres']
    # prediction['instance_segmentation_gt_meta']

    # panoptic instance prediction
    path = os.path.join(output_path, 'pred_path_panoptic_instance')
    mask_dir = 'predicted_masks'
    path_masks = os.path.join(path, mask_dir)
    os.makedirs(path, exist_ok=True)
    os.makedirs(path_masks, exist_ok=True)
    instance = get_fullres(prediction, 'panoptic_segmentation_deeplab_instance_idx').cpu().numpy()
    instance_meta = prediction['panoptic_segmentation_deeplab_instance_meta']

    for i, (instance_i, instance_meta_i) in enumerate(zip(instance,
                                                          instance_meta)):
        # write a txt file and corresponding masks for each example in batch
        basename = identifier_to_filename_mapper(batch['identifier'][i], ext='')

        txt_lines = []
        for instance_id in instance_meta_i:
            if instance_meta_i[instance_id]['area'] == 0:
                # empty instance (no offset was assigned to this center)
                continue

            # get mask ("everything non-zero is part of the prediction")
            mask = ((instance_i == instance_id)*255).astype(np.uint8)

            # save mask
            mask_fn = basename + f'_{len(txt_lines):03d}.png'
            cv2.imwrite(os.path.join(path_masks, mask_fn), mask)

            # prepare line for text file
            semantic_idx = semantic_class_mapper(
                instance_meta_i[instance_id]['semantic_idx']
            )
            panoptic_score = instance_meta_i[instance_id]['panoptic_score']
            txt_lines.append(
                f"{mask_dir}/{mask_fn} {semantic_idx} {panoptic_score:0.4f}\n"
            )

        with open(os.path.join(path, basename + '.txt'), 'w') as f:
            f.writelines(txt_lines)


def write_mapping_output(
    batch,
    prediction,
    output_path,
    instance_use_panoptic_score=True,
    semantic_class_mapper=lambda x: x,
    compressed=True
):
    # we only write predictions (see MIRA dataset readers in
    # nicr_scene_analysis_datasets for loading)

    def _write_as_npz(dirname, tensor_to_write):
        path = os.path.join(output_path, dirname)
        for i, tensor in enumerate(tensor_to_write):
            path_i = os.path.join(path, *batch['identifier'][i][:-1])
            filename_i = batch['identifier'][i][-1] + '.npz'
            os.makedirs(path_i, exist_ok=True)
            if compressed:
                np.savez_compressed(os.path.join(path_i, filename_i), tensor)
            else:
                np.savez(os.path.join(path_i, filename_i), tensor)

    # semantic prediction (float32: class + score)
    sem_scores = get_fullres(prediction, 'semantic_segmentation_score')
    sem_scores = torch.clamp(sem_scores, min=0, max=_SCORE_MAX)
    sem_classes = get_fullres(prediction, 'semantic_segmentation_idx')
    sem_classes = sem_classes.to(torch.uint8)    # < 255 classes
    sem_classes += 1     # 0 = void, but output has no void class -> +1
    sem_scores = sem_scores.cpu().numpy()
    sem_classes = sem_classes.cpu().numpy()
    sem_classes = semantic_class_mapper(sem_classes)    # map classes
    sem_output = sem_classes.astype('float32') + sem_scores
    assert (sem_output.astype('uint8') == sem_classes).all()

    # convert to topk format (topk, h, w) with topk=1 here for now
    sem_output = sem_output[:, None, ...]

    _write_as_npz('pred_semantic', sem_output)

    # panoptic semantic prediction (float32: class + score)
    # note panoptic merging is done on CPU
    pan_sem_scores = get_fullres(
        prediction,
        'panoptic_segmentation_deeplab_semantic_score'
    )
    pan_sem_scores = torch.clamp(pan_sem_scores, min=0, max=_SCORE_MAX)
    pan_sem_classes = get_fullres(prediction, 'panoptic_segmentation_deeplab_semantic_idx')
    pan_sem_classes = pan_sem_classes.to(torch.uint8)    # < 255 classes
    pan_sem_scores = pan_sem_scores.cpu().numpy()
    pan_sem_classes = pan_sem_classes.cpu().numpy()
    pan_sem_classes = semantic_class_mapper(pan_sem_classes)    # map classes
    pan_sem_output = pan_sem_classes.astype('float32') + pan_sem_scores
    assert (pan_sem_output.astype('uint8') == pan_sem_classes).all()

    # convert to topk format (topk, h, w) with topk=1
    pan_sem_output = pan_sem_output[:, None, ...]

    _write_as_npz('pred_panoptic_semantic', pan_sem_output)

    # panoptic instance prediction
    if instance_use_panoptic_score:
        # use panoptic score instead of instance score
        # score: score_instance_center * (mean_semantic_score_of_instance)
        pan_ins_scores = get_fullres(
            prediction,
            'panoptic_segmentation_deeplab_panoptic_score'
        )
    else:
        # use raw instance score
        # score: score_instance_center
        pan_ins_scores = get_fullres(
            prediction,
            'panoptic_segmentation_deeplab_instance_score'
        )
    pan_ins_scores = torch.clamp(pan_ins_scores, min=0, max=_SCORE_MAX)
    pan_ins_ids = get_fullres(prediction, 'panoptic_segmentation_deeplab_instance_idx')
    pan_ins_scores = pan_ins_scores.cpu().numpy()
    pan_ins_ids = pan_ins_ids.cpu().numpy()
    pan_ins_output = pan_ins_ids.astype('float32') + pan_ins_scores
    _write_as_npz('pred_panoptic_instance', pan_ins_output)

    # panoptic instance meta
    pan_ins_meta = prediction['panoptic_segmentation_deeplab_instance_meta']
    path = os.path.join(output_path, 'pred_panoptic_instance_meta')
    for i, meta in enumerate(pan_ins_meta):
        # apply semantic class mapping
        meta_i = deepcopy(meta)    # copy to be avoid to modify inplace
        for k in meta_i:
            if 'semantic_idx' in meta_i[k]:  # filter instances without pixels
                meta_i[k]['semantic_idx'] = int(semantic_class_mapper(
                    meta_i[k]['semantic_idx'])
                )
        path_i = os.path.join(path, *batch['identifier'][i][:-1])
        filename_i = batch['identifier'][i][-1] + '.json'
        os.makedirs(path_i, exist_ok=True)
        with open(os.path.join(path_i, filename_i), 'w') as f:
            json.dump(meta_i, f, sort_keys=True, indent=4)

    # TODO: when required: panoptic instance orientation

    # scene class prediction
    scene_scores = prediction['scene_class_score']
    scene_scores = torch.clamp(scene_scores, min=0, max=_SCORE_MAX)
    scene_classes = prediction['scene_class_idx']
    scene_scores = scene_scores.cpu().numpy()
    scene_classes = scene_classes.cpu().numpy()
    scene_output = scene_classes.astype('float32') + scene_scores
    _write_as_npz('pred_scene', scene_output)


def main():
    # args
    args = _get_args()

    if any(k in args.inference_output_format
           for k in ('scannet-semantic', 'scannet-instance',
                     'scannet-panoptic')):
        # ensure correct subsampling for ScanNet test split
        if 'scannet' == args.dataset and 'test' == args.inference_split:
            assert args.validation_scannet_subsample == 100

    # output path(s)
    if args.inference_output_path is None:
        # use weights path
        path, fn = os.path.split(args.weights_filepath)
        dir_name = f'inference_outputs_{os.path.splitext(fn)[0]}'
        args.inference_output_path = os.path.join(
            path,
            dir_name,
            args.dataset,
            args.inference_split
        )

    print(f"Writing inference outputs to: '{args.inference_output_path}'")
    os.makedirs(args.inference_output_path, exist_ok=True)

    # device
    device = torch.device('cuda')

    # data ---------------------------------------------------------------------
    # note that args.validation_scannet_subsample is used for ScanNet in test
    # phase, thus we overwrite it with args.inference_scannet_subsample
    args.validation_scannet_subsample = args.inference_scannet_subsample
    dataset = get_dataset(args, split=args.inference_split)

    # split dataset by camera -> batches of same spatial resolution
    datasets = tuple(
        deepcopy(dataset).filter_camera(camera)
        for camera in dataset.cameras
    )

    # build and set preprocessor
    preprocessor = get_preprocessor(
        args,
        dataset=dataset,
        phase='test',
        multiscale_downscales=None,
        keep_raw_inputs=True
    )
    for ds in datasets:
        ds.preprocessor = preprocessor

    # create dataloaders
    collate_fn = partial(
        mt_collate,
        type_blacklist=(np.ndarray, CollateIgnoredDict, OrientationDict,
                        SampleIdentifier)
    )
    dataloaders = tuple(
        DataLoader(
            ds,
            batch_size=args.inference_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=args.n_workers,
            persistent_workers=False
        )
        for ds in datasets
    )

    # max depth (parameter is given in m but we need it in mm)
    max_depth = None
    if args.inference_output_ground_truth_max_depth is not None:
        if 'scannet' != args.dataset:
            max_depth = args.inference_output_ground_truth_max_depth * 1000.0
        else:
            # Currently, we are using the depth image before preprocessing for
            # masking ground-truth annotations based on depth, as only before
            # preprocessing depth is in mm. However, for ScanNet, depth and RGB
            # are not registered and, thus, shapes may be different. As the
            # maximum depth is 10m for ScanNet, we simply disable the masking
            # for now.
            warnings.warn(
                "Masking ground-truth annotations based on "
                "`--inference-output-ground-truth-max-depth` disabled as "
                "dataset is ScanNet. Maximum distance is 10m."
            )

    # semantic class mapping --------------------------------------------------
    # ScanNet dataset only
    if args.dataset == 'scannet' and 20 == args.scannet_semantic_n_classes:
        mapping = ScanNet.SEMANTIC_CLASSES_20_MAPPING_TO_BENCHMARK  # with void
        mapping = np.array(list(mapping.values()), dtype=np.uint8)
        semantic_class_mapper = lambda x: mapping[x]
    else:
        semantic_class_mapper = lambda x: x

    # identifier mapping ------------------------------------------------------
    # scannet-* output format only
    if 'scannet' == args.dataset:

        def _identifier_to_filename(identifier, ext='.png'):
            # format scene%04d_%02d_%06d.png
            camera, scene, id_ = identifier
            return f'{scene}_{int(id_):06d}{ext}'

    elif 'hypersim' == args.dataset:

        def _identifier_to_filename(identifier, ext='.png'):
            # format scene_camera%04d_%02d_%06d.png
            scene, camera, id_ = identifier
            return f'{scene}_{camera}_{int(id_):06d}{ext}'

    else:
        raise RuntimeError()

    # model -------------------------------------------------------------------
    model = EMSANet(args, dataset_config=dataset.config)

    # load weights
    print(f"Loading checkpoint: '{args.weights_filepath}'.")
    checkpoint = torch.load(args.weights_filepath, map_location='cpu')
    if 'epoch' in checkpoint:
        print(f"-> Epoch: {checkpoint['epoch']}")
    if args.debug and 'logs' in checkpoint:
        print(f"-> Logs/Metrics:")
        pprint(checkpoint['logs'])

    state_dict = checkpoint['state_dict']
    load_weights(args, model, state_dict)

    # set model to eval mode
    torch.set_grad_enabled(False)
    model.eval()
    model.to(device)

    # inference ---------------------------------------------------------------
    # write some meta data
    ts = time()
    meta = {
        'command': ' '.join(sys.argv),
        'args': vars(args),
        'timestamp': int(ts),
        'local_time': datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'),
        'user': getpass.getuser(),
        'checkpoint': {}
    }
    if 'epoch' in checkpoint:
        meta['checkpoint']['epoch'] = checkpoint['epoch']
    if 'logs' in checkpoint:
        meta['checkpoint']['logs'] = {k: v.item() if torch.is_tensor(v) else v
                                      for k, v in checkpoint['logs'].items()}

    fp = os.path.join(args.inference_output_path, 'meta.json')
    meta_list = []
    # check for existing meta information
    if os.path.exists(fp):
        with open(fp, 'r') as f:
            meta_list = json.load(f)
    # write meta information
    meta_list.append(meta)
    with open(fp, 'w') as f:
        json.dump(meta_list, f, indent=4)

    # determine max instances per category (class)
    panoptic_post = model.decoders['panoptic_helper'].postprocessing
    max_instances_per_category = panoptic_post.max_instances_per_category

    # run inference and write outputs
    for i, dataloader in enumerate(dataloaders):
        camera = dataloader.dataset.camera

        for j, batch in tqdm(enumerate(dataloader),
                             total=len(dataloader),
                             desc=f'{i+1}/{len(dataloaders)} ({camera})'):
            # move batch to device
            batch = move_batch_to_device(batch, device=device)

            # apply model
            prediction = model(batch, do_postprocessing=True)

            # write outputs
            for output_format in args.inference_output_format:
                # determine and create output path if not exists
                output_path = os.path.join(
                    args.inference_output_path,
                    output_format.replace('-', '_'),
                )
                os.makedirs(output_path,
                            exist_ok=(args.overwrite or j != 0 or i != 0))

                if 'scannet-semantic' == output_format:
                    write_scannet_semantic_output(
                        batch=batch,
                        prediction=prediction,
                        output_path=output_path,
                        identifier_to_filename_mapper=_identifier_to_filename,
                        max_depth=max_depth,
                        semantic_class_mapper=semantic_class_mapper,
                        write_gt=args.inference_output_write_ground_truth
                    )
                elif 'scannet-instance' == output_format:
                    write_scannet_instance_output(
                        batch=batch,
                        prediction=prediction,
                        output_path=output_path,
                        identifier_to_filename_mapper=_identifier_to_filename,
                        shift=args.inference_output_semantic_instance_shift,
                        max_depth=max_depth,
                        semantic_class_mapper=semantic_class_mapper,
                        write_gt=args.inference_output_write_ground_truth
                    )
                elif 'scannet-panoptic' == output_format:
                    write_scannet_panoptic_output(
                        batch=batch,
                        prediction=prediction,
                        output_path=output_path,
                        max_instances_per_category=max_instances_per_category,
                        identifier_to_filename_mapper=_identifier_to_filename,
                        max_depth=max_depth,
                        semantic_class_mapper=semantic_class_mapper,
                        write_gt=args.inference_output_write_ground_truth
                    )
                elif 'mapping' == output_format:
                    write_mapping_output(
                        batch=batch,
                        prediction=prediction,
                        output_path=output_path,
                        instance_use_panoptic_score=True,
                        semantic_class_mapper=semantic_class_mapper,
                        compressed=True
                    )


if __name__ == '__main__':
    main()
