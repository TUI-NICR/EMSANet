# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
"""
from typing import Any, Dict

import os
import PIL

import numpy as np

from nicr_mt_scene_analysis.data.preprocessing.clone import DEFAULT_CLONE_KEY
from nicr_mt_scene_analysis.types import BatchType
from nicr_mt_scene_analysis.visualization.semantic import visualize_semantic_pil
from nicr_mt_scene_analysis.visualization.instance import visualize_instance_pil
from nicr_mt_scene_analysis.visualization.instance import visualize_instance
from nicr_mt_scene_analysis.visualization.instance import visualize_orientation_pil
from nicr_mt_scene_analysis.visualization.instance import visualize_instance_orientations_pil
from nicr_mt_scene_analysis.visualization.instance import visualize_instance_orientations
from nicr_mt_scene_analysis.visualization.instance import visualize_instance_center_pil
from nicr_mt_scene_analysis.visualization.instance import visualize_instance_offset_pil
from nicr_mt_scene_analysis.visualization.panoptic import visualize_panoptic_pil
from nicr_mt_scene_analysis.visualization.depth import visualize_depth_pil
from nicr_mt_scene_analysis.visualization import to_pil_img
from nicr_mt_scene_analysis.utils.panoptic_merge import deeplab_merge_batch
from nicr_scene_analysis_datasets.dataset_base import DatasetConfig


def save_visualization_result_dict(
    visualization_dict: Dict[str, Any],
    output_dir: str
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for key, value in visualization_dict.items():
        if key == 'identifier':
            continue
        for i, v in enumerate(value):
            out_filepath = os.path.join(
                output_dir,
                key,
                *visualization_dict['identifier'][i]
            )
            os.makedirs(os.path.dirname(out_filepath), exist_ok=True)

            if isinstance(v, PIL.Image.Image):
                v.save(out_filepath + '.png')
            else:
                # scene label
                with open(out_filepath + '.txt', 'w') as f:
                    f.write(str(v))


def blend_images(
    img1: np.ndarray,
    img2: np.ndarray,
    alpha: float = 0.2
) -> np.ndarray:
    # ensure that img is a numpy object
    img1 = np.asanyarray(img1)
    img2 = np.asanyarray(img2)
    assert img1.dtype == img2.dtype
    assert img1.ndim == img2.ndim

    # ensure that img is a numpy object
    img1 = np.asanyarray(img1)
    img2 = np.asanyarray(img2)

    # alpha composite images
    if img2.ndim == 3:
        mask = np.any(img2 > 0, axis=2)
    else:
        mask = img2 > 0
    result = img1.copy()
    result[mask, ...] = (
        (1-alpha)*img1[mask, ...] + alpha*img2[mask, ...]
    ).astype(img1.dtype)

    return result


def visualize_batches(
    batch: BatchType,
    dataset_config: DatasetConfig
) -> Dict[str, Any]:
    colors = dataset_config.semantic_label_list.colors_array
    kwargs_instance_orientation = {
        'thickness': 3,
        'font_size': 45,
        'bg_color': 255,
        'bg_color_font': 'white'
    }
    result_dict = {}
    result_dict['identifier'] = batch['identifier']

    if DEFAULT_CLONE_KEY in batch:
        batch_np = batch[DEFAULT_CLONE_KEY]
        if 'rgb' in batch_np:
            result_dict['rgb'] = [
                to_pil_img(img) for img in batch_np['rgb']
            ]

        if 'depth' in batch_np:
            result_dict['depth'] = [
                visualize_depth_pil(img) for img in batch_np['depth']
            ]

        if 'semantic' in batch_np:
            result_dict['semantic'] = [
                visualize_semantic_pil(img, colors=colors)
                for img in batch_np['semantic']
            ]
    else:
        # we do not have the batch data without preprocessing
        batch_np = {}

    # instance -----------------------------------------------------------------
    result_instance = []
    if 'instance' in batch:
        for img, fg in zip(batch['instance'], batch['instance_foreground']):
            img = visualize_instance(img.cpu().numpy())
            img[~fg.cpu().numpy()] = [255, 255, 255]
            result_instance.append(to_pil_img(img))
        result_dict['instance'] = result_instance

        result_dict['instance_center'] = [
            visualize_instance_center_pil(img.cpu().numpy())
            for img in batch['instance_center']
        ]

        result_dict['instance_offset'] = [
            visualize_instance_offset_pil(
                img.permute(1, 2, 0).cpu().numpy(),
                fg.cpu().numpy()
            ) for img, fg in zip(batch['instance_offset'],
                                 batch['instance_foreground'])
        ]

    # orientation --------------------------------------------------------------
    if DEFAULT_CLONE_KEY in batch and 'orientations' in batch_np:
        result_dict['instance_orientation'] = [
            visualize_instance_orientations_pil(
                *data,
                **kwargs_instance_orientation
            ) for data in zip(batch_np['instance'],
                              batch_np['orientations'])
        ]

    if 'orientation' in batch:
        results = []
        for orientation, fg in zip(batch['orientation'],
                                   batch['orientation_foreground']):
            orientation_img = visualize_orientation_pil(
                orientation.permute(1, 2, 0).cpu().numpy()
            )
            orientation_img = np.array(orientation_img)
            orientation_img[~fg.cpu().numpy()] = [255, 255, 255]
            results.append(to_pil_img(orientation_img))

        result_dict['orientations_masked'] = results

        result_dict['orientations'] = [
            visualize_orientation_pil(
                img.permute(1, 2, 0).cpu().numpy()
            ) for img in batch['orientation']
        ]

    # panoptic -----------------------------------------------------------------
    if all(x in batch for x in ('semantic_fullres', 'instance_fullres')):
        categories = []
        for idx, label in enumerate(dataset_config.semantic_label_list):
            label_dict = {}
            label_dict['supercategory'] = label.class_name
            label_dict['name'] = label.class_name
            label_dict['id'] = idx
            label_dict['isthing'] = int(label.is_thing)
            label_dict['color'] = colors[idx]
            categories.append(label_dict)
        categories = {cat['id']: cat for cat in categories}
        kwargs_panoptic = {
            'num_cats': len(dataset_config.semantic_label_list),
            'categorys': categories,
            'max_instances': 1 << 16,
        }
        panoptic_targets, _ = deeplab_merge_batch(
            semantic_batch=batch['semantic_fullres'],
            instance_batch=batch['instance_fullres'],
            instance_fg_batch=[img != 0 for img in batch['instance_fullres']],
            max_instances_per_category=1 << 16,
            thing_ids=np.where(dataset_config.semantic_label_list.classes_is_thing)[0],
            void_label=0
        )

        result_dict['panoptic'] = [
            visualize_panoptic_pil(img.cpu().numpy(), **kwargs_panoptic)
            for img in panoptic_targets
        ]

    # panoptic + orientation ---------------------------------------------------
    if all((DEFAULT_CLONE_KEY in batch, 'instance' in batch_np,
            'orientations' in batch, 'panoptic' in result_dict)):
        result_list = []
        for instance, orientation_pred, panoptic in zip(
            batch_np['instance'],
            batch['orientations'],
            result_dict['panoptic']
        ):
            orientation = visualize_instance_orientations(
                instance,
                orientation_pred,
                draw_outline=False,
                thickness=3,
                font_size=45,
                bg_color=0,
                bg_color_font='black'
            )

            panoptic_orientation = np.array(panoptic.copy())
            panoptic_orientation[(orientation != 0)] = 255
            panoptic_orientation = to_pil_img(panoptic_orientation)
            result_list.append(panoptic_orientation)
        result_dict['panoptic_orientation'] = result_list

    # scene classification -----------------------------------------------------
    if DEFAULT_CLONE_KEY in batch and 'scene' in batch_np:
        result_dict['scene'] = [
            dataset_config.scene_label_list[s].class_name
            for s in batch_np['scene']
        ]

    return result_dict


def visualize_predictions(
    predictions: Dict[str, Any],
    batch: BatchType,
    dataset_config: DatasetConfig
) -> Dict[str, Any]:
    colors = dataset_config.semantic_label_list.colors_array

    result_dict = {}
    result_dict['identifier'] = batch['identifier']

    # semantic segmentation ----------------------------------------------------
    if 'semantic_segmentation_idx_fullres' in predictions:
        result_dict['semantic'] = [
            visualize_semantic_pil(img.cpu().numpy(), colors=colors[1:])
            for img in predictions['semantic_segmentation_idx_fullres']
        ]

    # instance segmentation ----------------------------------------------------
    if 'instance_segmentation_gt_foreground_fullres' in predictions:
        result_dict['instance_gt_fg'] = [
            visualize_instance_pil(img.cpu().numpy())
            for img in predictions['instance_segmentation_gt_foreground_fullres']
        ]

    if 'instance_centers' in predictions:
        result_dict['instance_center'] = [
            visualize_instance_center_pil(img[0].cpu().numpy())
            for img in predictions['instance_centers']
        ]

    if 'instance_offsets' in predictions:
        result_dict['instance_offset'] = [
            visualize_instance_offset_pil(img.permute(1, 2, 0).cpu().numpy())
            for img in predictions['instance_offsets']
        ]

        if 'panoptic_foreground_mask' in predictions:
            result_dict['instance_offset_masked'] = [
                visualize_instance_offset_pil(
                    img.permute(1, 2, 0).cpu().numpy(),
                    fg.cpu().numpy()
                ) for img, fg in zip(predictions['instance_offsets'],
                                     predictions['panoptic_foreground_mask'])
            ]

    # panoptic segmentation ----------------------------------------------------
    categories = []
    for idx, label in enumerate(dataset_config.semantic_label_list):
        label_dict = {}
        label_dict['supercategory'] = label.class_name
        label_dict['name'] = label.class_name
        label_dict['id'] = idx
        label_dict['isthing'] = int(label.is_thing)
        label_dict['color'] = colors[idx]
        categories.append(label_dict)
    categories = {cat['id']: cat for cat in categories}

    if 'panoptic_segmentation_deeplab_fullres' in predictions:
        result_dict['panoptic'] = [
            visualize_panoptic_pil(
                img.cpu().numpy(),
                num_cats=len(dataset_config.semantic_label_list),
                categorys=categories,
                max_instances=1 << 16)
            for img in predictions['panoptic_segmentation_deeplab_fullres']
        ]

    if 'panoptic_instance_segmentation_fullres' in predictions:
        instance = predictions['panoptic_instance_segmentation_fullres']
        result_dict['instance'] = [
            visualize_instance_pil(img.cpu().numpy())
            for img in instance
        ]

    # orientation estimation ---------------------------------------------------
    kwargs_instance_orientation = {'thickness': 3,
                                   'font_size': 45,
                                   'bg_color': 255,
                                   'bg_color_font': 'white'}

    if 'orientations_gt_instance_gt_orientation_foreground' in predictions:
        result_dict['instance_orientation_gt_fg'] = [
            visualize_instance_orientations_pil(*data, **kwargs_instance_orientation)
            for data in zip(batch['instance'].cpu().numpy(),
                            predictions['orientations_gt_instance_gt_orientation_foreground'])
        ]

    if 'orientations_panoptic_segmentation_deeplab_instance_segmentation' in predictions:
        sem_seg_no_void = predictions['semantic_segmentation_idx_fullres']
        use_orienation_idx = np.where(
            dataset_config.semantic_label_list_without_void.classes_use_orientations)[
            0]
        orientation_fg = np.isin(sem_seg_no_void.cpu().numpy(),
                                 use_orienation_idx)
        instance_img = predictions['panoptic_instance_segmentation_fullres'].cpu().numpy()
        instance_img[~orientation_fg] = 0
        result_dict['instance_orientation'] = [visualize_instance_orientations_pil(
            *data,
            **kwargs_instance_orientation
        ) for data in zip(instance_img,
                          predictions['orientations_panoptic_segmentation_deeplab_instance_segmentation'])
        ]

    if 'instance_orientation' in predictions:
        result_dict['orientations'] = [
            visualize_orientation_pil(img.permute(1, 2, 0).cpu().numpy())
            for img in predictions['instance_orientation']
        ]

    # everything combined ------------------------------------------------------
    if all(('orientations_panoptic_segmentation_deeplab_instance_segmentation' in predictions,
            DEFAULT_CLONE_KEY in batch)):
        result_panoptic_orientation_list = []
        result_panoptic_orientation_rgb_list = []
        result_instance_orientation_list = []
        result_orientation_list = []
        for instance, instance_vis, orientation_pred, panoptic, rgb in zip(
            instance_img,
            result_dict['instance'],
            predictions['orientations_panoptic_segmentation_deeplab_instance_segmentation'],
            result_dict['panoptic'],
            batch[DEFAULT_CLONE_KEY]['rgb']
        ):

            orientation = visualize_instance_orientations(
                instance,
                orientation_pred,
                draw_outline=False,
                thickness=3,
                font_size=45,
                bg_color=0,
                bg_color_font='black'
            )

            panoptic_orientation = np.array(panoptic.copy())
            panoptic_orientation[(orientation != 0)] = 255
            panoptic_orientation_rgb = to_pil_img(
                blend_images(panoptic_orientation, rgb, alpha=0.5)
            )
            panoptic_orientation = to_pil_img(panoptic_orientation)
            result_panoptic_orientation_list.append(panoptic_orientation)
            result_panoptic_orientation_rgb_list.append(panoptic_orientation_rgb)

            instance_orientation = np.array(instance_vis.convert())
            instance_orientation[(orientation != 0)] = 255
            instance_orientation = to_pil_img(instance_orientation)
            result_instance_orientation_list.append(instance_orientation)

            result_orientation_list.append(to_pil_img(orientation))

        result_dict['panoptic_orientation'] = result_panoptic_orientation_list
        result_dict['panoptic_orientation_rgb'] = result_panoptic_orientation_rgb_list
        result_dict['instance_orientation'] = result_instance_orientation_list
        result_dict['panoptic_orientation_mask'] = result_orientation_list

    # scene classification -----------------------------------------------------
    if 'scene_class_idx' in predictions:
        result_dict['scene'] = [
            dataset_config.scene_label_list_without_void[s].class_name
            for s in predictions['scene_class_idx']
        ]

    return result_dict


def visualize(
    output_path: str,
    batch: BatchType,
    predictions: Dict[str, Any],
    dataset_config: DatasetConfig
) -> None:
    # visualize ground truth
    gt_path = os.path.join(output_path, 'gt')
    batch_visualization = visualize_batches(
        batch=batch,
        dataset_config=dataset_config
    )
    save_visualization_result_dict(
        visualization_dict=batch_visualization,
        output_dir=gt_path
    )

    # visualize ground truth for side outputs (downscaled images)
    additional_keys = ['_down_8', '_down_16', '_down_32']
    for key in additional_keys:
        if key not in batch:
            continue
        r_path = os.path.join(gt_path, key)
        n_batch = batch[key]
        n_batch['identifier'] = batch['identifier']
        n_batch_visualization = visualize_batches(
            batch=n_batch,
            dataset_config=dataset_config
        )
        save_visualization_result_dict(
            visualization_dict=n_batch_visualization,
            output_dir=r_path
        )

    # visualize predictions
    prediction_visualization = visualize_predictions(
        predictions=predictions,
        batch=batch,
        dataset_config=dataset_config
    )
    save_visualization_result_dict(
        visualization_dict=prediction_visualization,
        output_dir=os.path.join(output_path, 'pred')
    )
