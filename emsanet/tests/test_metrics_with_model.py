# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
import json
import os

import torch
import numpy as np
import pytest
import PIL.Image as Image
from tqdm import tqdm

from nicr_mt_scene_analysis import metric
from nicr_mt_scene_analysis.data import move_batch_to_device
from nicr_mt_scene_analysis.data.preprocessing.resize import get_fullres
from nicr_mt_scene_analysis.utils.panoptic_merge import deeplab_merge_batch
from nicr_scene_analysis_datasets.utils.testing import DATASET_PATH_DICT
from panopticapi.utils import IdGenerator
from panopticapi.evaluation import pq_compute

from emsanet.args import ArgParserEMSANet
from emsanet.data import get_datahelper
from emsanet.model import EMSANet
from emsanet.preprocessing import get_preprocessor


PQ_TEST_WEIGHT_DICT = {
    'nyuv2': '/results_nas/emsanet/_final_submission/_selected_runs_paper/results_emsanet_mt_semantic+scene+instance+orientation/nyuv2/run_2022_05_07-20_24_44-721813/checkpoints/ckpt_valid_panoptic_all_deeplab_pq_best.pth',
    'sunrgbd': '/results_nas/emsanet/_final_submission/_selected_runs_paper/results_emsanet_mt_semantic+scene+instance+orientation/sunrgbd/run_2022_05_05-10_57_16-340081/checkpoints/ckpt_valid_panoptic_all_deeplab_pq_epoch_0453.pth'
    # 'sunrgbd': '/results_nas/emsanet/results_mt_semantic+scene+instance+orientation/sunrgbd/run_2022_02_11-13_51_17-488749/checkpoints/ckpt_valid_semantic_miou_best.pth'
}


@pytest.mark.parametrize('dataset', ('nyuv2', 'sunrgbd'))
def test_compare_pq_with_panopticapi(tmp_path, dataset):
    """Test that compares our pq to the panopticapi"""
    parser = ArgParserEMSANet()
    args = parser.parse_args(['--dataset', dataset], verbose=False)
    args.wandb_mode = 'disabled'
    args.rgb_encoder_backbone = 'resnet34'
    args.depth_encoder_backbone_block = 'nonbottleneck1d'
    args.input_modalities = ('rgb', 'depth')
    args.tasks = ('instance', 'semantic', 'scene', 'orientation')
    args.enable_panoptic = True
    args.no_pretrained_backbone = True
    args.dataset_path = DATASET_PATH_DICT[args.dataset]
    args.validation_batch_size = 4
    label_divisor = (1 << 16)

    data = get_datahelper(args)
    dataset_config = data.dataset_config
    n_semantic_classes = len(dataset_config.semantic_label_list)
    is_thing = dataset_config.semantic_label_list.classes_is_thing

    # we process only 25 batches of each camera to speed up testing, use
    # -1 or 0 to process all batches of a camera
    batches_per_cam = 25

    model = EMSANet(args, dataset_config=data.dataset_config)
    model = model.to(torch.device('cpu'))

    checkpoint = torch.load(PQ_TEST_WEIGHT_DICT[dataset])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()

    # set preprocessor to datasets (note, preprocessing depends on model)
    downscales = set()
    for decoder in model.decoders.values():
        downscales |= set(decoder.side_output_downscales)

    data.set_valid_preprocessor(
        get_preprocessor(
            args,
            dataset=data.datasets_valid[0],
            phase='test',
            multiscale_downscales=tuple(downscales) if args.debug else None
        )
    )

    pq_metric = metric.PanopticQuality(
        num_categories=n_semantic_classes,
        ignored_label=0,
        max_instances_per_category=label_divisor,
        offset=256**3,
        is_thing=is_thing
    )

    gt_path = os.path.join(tmp_path, 'gt')
    os.makedirs(gt_path, exist_ok=True)
    pred_path = os.path.join(tmp_path, 'pred')
    os.makedirs(pred_path, exist_ok=True)

    img_ctr = 0
    categorys = []
    for idx, label in enumerate(dataset_config.semantic_label_list):
        label_dict = {}
        label_dict['supercategory'] = label.class_name
        label_dict['name'] = label.class_name
        label_dict['id'] = idx
        label_dict['isthing'] = int(label.is_thing)
        label_dict['color'] = [int(a) for a in label.color]
        categorys.append(label_dict)
    categorys_list = categorys.copy()
    categorys = {cat['id']: cat for cat in categorys}
    categorys_json_file = os.path.join(tmp_path, 'categories.json')
    with open(categorys_json_file, 'w') as f:
        json.dump(categorys, f)

    # Modified copy of the original code from
    # https://github.com/cocodataset/panopticapi/blob/master/converters/2channels2panoptic_coco_format.py#L38
    def convert(image_id, pan, categories, file_name, segmentations_folder,
                VOID=0, OFFSET=label_divisor):
        h, w = pan.shape
        pan_format = np.zeros((h, w, 3), dtype=np.uint8)

        id_generator = IdGenerator(categories)

        uids = np.unique(pan)
        segm_info = []
        for el in uids:
            sem = el // OFFSET
            if sem == VOID:
                continue
            if sem not in categories:
                raise KeyError('Unknown semantic label {}'.format(sem))
            mask = pan == el
            segment_id, color = id_generator.get_id_and_color(sem)
            pan_format[mask] = color
            segm_info.append({"id": segment_id,
                              "category_id": int(sem),
                              "iscrowd": 0,
                              "area": int(mask.sum())})

        annotation = {'image_id': image_id,
                      'file_name': file_name,
                      'segments_info': segm_info}

        Image.fromarray(pan_format).save(os.path.join(segmentations_folder, file_name))
        return annotation

    annotations_gt = []
    annotations_pred = []
    images = []
    for cam in data.valid_dataloaders:
        for idx, batch in tqdm(enumerate(cam), total=len(cam)):
            if batches_per_cam > 0 and idx > batches_per_cam:
                break
            batch = move_batch_to_device(batch, torch.device('cpu'))
            with torch.no_grad():
                output = model(batch, do_postprocessing=True)

            # batch = move_batch_to_device(batch, torch.device("cpu"))
            semantic_batch = get_fullres(batch, 'semantic')
            instance_batch = get_fullres(batch, 'instance')
            instance_fg = instance_batch != 0
            panoptic_targets, _ = deeplab_merge_batch(semantic_batch,
                                                      instance_batch,
                                                      instance_fg,
                                                      label_divisor,
                                                      np.where(is_thing)[0],
                                                      0)
            panoptic_preds = output['panoptic_segmentation_deeplab_fullres']
            panoptic_targets = panoptic_targets.cpu()
            panoptic_preds = panoptic_preds.cpu()
            pq_metric.update(panoptic_preds, panoptic_targets)

            for target_img, pred_img in zip(panoptic_targets, panoptic_preds):
                pan_img_name = f'{img_ctr:05d}.png'

                annotation = convert(img_ctr, target_img, categorys, pan_img_name,
                                     gt_path, VOID=0, OFFSET=label_divisor)
                annotations_gt.append(annotation)

                annotation = convert(img_ctr, pred_img, categorys, pan_img_name,
                                     pred_path, VOID=0, OFFSET=label_divisor)
                annotations_pred.append(annotation)

                image_dict = {}
                image_dict['file_name'] = pan_img_name
                image_dict['id'] = img_ctr
                images.append(image_dict)
                img_ctr += 1

    pq_result = pq_metric.compute()

    coco_dict_gt = {}
    coco_dict_gt['images'] = images
    coco_dict_gt['annotations'] = annotations_gt
    coco_dict_gt['categories'] = categorys_list
    coco_dict_pred = {}
    coco_dict_pred['images'] = images
    coco_dict_pred['annotations'] = annotations_pred
    coco_dict_pred['categories'] = categorys_list
    pan_gt_json_file = os.path.join(tmp_path, 'panoptic_gt.json')
    pan_pred_json_file = os.path.join(tmp_path, 'panoptic_pred.json')
    with open(pan_gt_json_file, 'w') as f:
        json.dump(coco_dict_gt, f)

    with open(pan_pred_json_file, 'w') as f:
        json.dump(coco_dict_pred, f)

    pq_coco_panoptic_api = pq_compute(
        pan_gt_json_file, pan_pred_json_file,
        gt_path, pred_path
    )

    np.testing.assert_almost_equal(pq_result['all_pq'],
                                   pq_coco_panoptic_api['All']['pq'],
                                   decimal=9)
    np.testing.assert_almost_equal(pq_result['all_sq'],
                                   pq_coco_panoptic_api['All']['sq'],
                                   decimal=9)
    np.testing.assert_almost_equal(pq_result['all_rq'],
                                   pq_coco_panoptic_api['All']['rq'],
                                   decimal=9)

    np.testing.assert_almost_equal(pq_result['things_pq'],
                                   pq_coco_panoptic_api['Things']['pq'],
                                   decimal=9)
    np.testing.assert_almost_equal(pq_result['things_sq'],
                                   pq_coco_panoptic_api['Things']['sq'],
                                   decimal=9)
    np.testing.assert_almost_equal(pq_result['things_rq'],
                                   pq_coco_panoptic_api['Things']['rq'],
                                   decimal=9)

    np.testing.assert_almost_equal(pq_result['stuff_pq'],
                                   pq_coco_panoptic_api['Stuff']['pq'],
                                   decimal=9)
    np.testing.assert_almost_equal(pq_result['stuff_sq'],
                                   pq_coco_panoptic_api['Stuff']['sq'],
                                   decimal=9)
    np.testing.assert_almost_equal(pq_result['stuff_rq'],
                                   pq_coco_panoptic_api['Stuff']['rq'],
                                   decimal=9)
