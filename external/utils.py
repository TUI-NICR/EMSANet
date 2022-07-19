# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
import os
import json

from detectron2.data import DatasetCatalog, MetadataCatalog

from PIL import Image

from nicr_scene_analysis_datasets import d2 as nicr_d2


def create_coco_pan_gt(dataset_name, output_folder_gt, output_pan_gt_json_file):
    os.makedirs(output_folder_gt)
    dataset = DatasetCatalog.get(dataset_name)
    dataset_meta = MetadataCatalog.get(dataset_name)
    dataset_config = dataset_meta.dataset_config
    nicr_mapper = nicr_d2.NICRSceneAnalysisDatasetMapper(dataset_config)
    annotations = []
    images = []
    for sample in dataset:
        sample = nicr_mapper(sample)
        file_name = sample['file_name']
        pan_seg = sample['pan_seg']
        output_file = os.path.join(output_folder_gt, file_name)
        Image.fromarray(pan_seg).save(output_file, format="PNG")

        image_dict = {}
        image_dict['file_name'] = file_name
        image_dict['id'] = sample['image_id']
        images.append(image_dict)

        annotations.append({
            'image_id': sample['image_id'],
            'file_name': file_name,
            'segments_info': sample['segments_info_json']
        })

    coco_dict_gt = {}
    coco_dict_gt['images'] = images
    coco_dict_gt['annotations'] = annotations
    coco_dict_gt['categories'] = nicr_mapper.categories_list
    with open(output_pan_gt_json_file, 'w') as f:
        json.dump(coco_dict_gt, f)
