import os
import sys

import cv2

import numpy as np
from nicr_scene_analysis_datasets import d2
from detectron2.data import MetadataCatalog

from panopticapi.utils import rgb2id

from PIL import Image
from tqdm import tqdm
import json


def main():
    # Parse first argument as path to pred images
    # if len(sys.argv) < 2:
    #     raise ValueError("Please provide path to pred images")

    # images_path = sys.argv[1]
    images_path = '/home/sofi9432/Desktop/multi-task-scene-analysis/output/inference/nyuv2_test/pan_pred'
    images = os.listdir(images_path)

    dataset_name = 'nyuv2_test'
    dataset_meta = MetadataCatalog.get(dataset_name)

    sem_output_dir = os.path.join(images_path, 'sem_seg')
    os.makedirs(sem_output_dir, exist_ok=True)
    pan_output_dir = os.path.join(images_path, 'pan_seg_mod')
    os.makedirs(pan_output_dir, exist_ok=True)

    label_divisor = 256

    dataset_meta = MetadataCatalog.get(dataset_name)
    dataset_config = dataset_meta.dataset_config
    mapper = d2.NICRSceneAnalysisDatasetMapper(dataset_config)
    semantic_colors = dataset_meta.stuff_colors

    annotations = []
    images_json = []
    for image in tqdm(images):
        # img = cv2.imread(os.path.join(images_path, image), cv2.IMREAD_UNCHANGED)
        if '.png' in image:
            img = Image.open(os.path.join(images_path, image))
        else:
            continue
        img = np.array(img)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = rgb2id(img)
        sem_seg = img // label_divisor
        # instance_seg = img % label_divisor
        # Take one channel sem seg and convert it to three channel image
        sem_seg_color = np.zeros((sem_seg.shape[0], sem_seg.shape[1], 3), dtype=np.uint8)
        for idx in np.unique(sem_seg):
            sem_seg_color[sem_seg == idx] = semantic_colors[idx]
        # sem_seg_color = cv2.cvtColor(sem_seg_color, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(os.path.join(sem_output_dir, image), sem_seg_color)
        Image.fromarray(sem_seg_color).save(os.path.join(sem_output_dir, image), format="PNG")

        # Fix void
        sem_seg += 1
        dataset_dict = {
            'instance': img,
            'semantic': sem_seg,
            'rgb': np.zeros((sem_seg.shape[0], sem_seg.shape[1], 3), dtype=np.uint8),
            'identifier': image
        }
        mapped_data = mapper(dataset_dict)
        pan_seg = mapped_data['pan_seg']
        # cv2.imwrite(os.path.join(pan_output_dir, image), pan_seg)
        Image.fromarray(pan_seg).save(os.path.join(pan_output_dir, image), format="PNG")

        image_dict = {}
        image_dict['file_name'] = image
        image_dict['id'] = image.split('.')[0]
        images_json.append(image_dict)

        annotations.append({
            'image_id': image.split('.')[0],
            'file_name': image,
            'segments_info': mapped_data['segments_info']
        })

    coco_dict_gt = {}
    coco_dict_gt['images'] = images
    coco_dict_gt['annotations'] = annotations
    coco_dict_gt['categories'] = mapper.categories_list
    with open(os.path.join(pan_output_dir, 'pred.json'), 'w') as f:
        json.dump(coco_dict_gt, f)
        # print(1)


if __name__ == "__main__":
    main()
