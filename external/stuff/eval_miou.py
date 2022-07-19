import torch
from builtins import enumerate, print
import os
import sys

import cv2
import numpy as np

from nicr_scene_analysis_datasets import d2 as nicr_d2
from detectron2.data import MetadataCatalog
from mt_scene_analysis.metric import MeanIntersectionOverUnion
from nicr_scene_analysis_datasets.pytorch import NYUv2


def main():
    # Get path to the predicted semantic segmentation results
    # if len(sys.argv) < 2:
    #     raise ValueError("Please provide the path to the predicted semantic segmentation results")
    # pred_path = sys.argv[1]
    pred_path = "/home/sofi9432/Desktop/multi-task-scene-analysis/epoch_497"
    output_dir = "/home/sofi9432/Desktop/multi-task-scene-analysis/epoch_497_colored"
    nyuv2_path = "/datasets_nas/nicr_scene_analysis_datasets/version_040/nyuv2"

    nyuv2 = NYUv2(
        dataset_path=nyuv2_path,
        split="test",
        sample_keys=(
            'identifier',
            'semantic')
    )

    os.makedirs(output_dir, exist_ok=True)

    classes_without_void = MetadataCatalog.get('nyuv2_test').stuff_classes
    colors_without_void = MetadataCatalog.get('nyuv2_test').stuff_colors

    # +1 cause of void
    n_classes = len(colors_without_void) + 1
    miou_metric = MeanIntersectionOverUnion(n_classes, ignore_first_class=True)

    for gt in nyuv2:
        identifier = gt['identifier'][0]
        gt_sem_seg = gt['semantic']

        prediction_file = identifier + '.png'
        prediction_path = os.path.join(pred_path, prediction_file)
        pred_sem_seg = cv2.imread(prediction_path, cv2.IMREAD_UNCHANGED)
        pred_sem_seg_colored = np.zeros(pred_sem_seg.shape + (3,),
                                        dtype=np.uint8)

        for idx, color in enumerate(colors_without_void):
            pred_sem_seg_colored[pred_sem_seg == idx] = color

        pred_sem_seg_colored = cv2.cvtColor(pred_sem_seg_colored,
                                            cv2.COLOR_RGB2BGR)
        output_path = os.path.join(output_dir, prediction_file)
        cv2.imwrite(output_path, pred_sem_seg_colored)

        # For mIoU computation including void
        pred_sem_seg += 1

        gt_sem_seg = torch.from_numpy(gt_sem_seg).long()
        pred_sem_seg = torch.from_numpy(pred_sem_seg).long()

        miou_metric.update(pred_sem_seg, gt_sem_seg)

    print(f'mIoU: {miou_metric.compute():.4f}')


if __name__ == '__main__':
    main()
