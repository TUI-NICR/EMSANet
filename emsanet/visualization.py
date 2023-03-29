# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
from typing import Any, Dict, Optional, Sequence, Union

import os
import warnings

import cv2
import numpy as np
import PIL

from nicr_mt_scene_analysis.data.preprocessing.clone import DEFAULT_CLONE_KEY
from nicr_mt_scene_analysis.data.preprocessing.resize import get_fullres_key
from nicr_mt_scene_analysis.types import BatchType
from nicr_mt_scene_analysis.visualization import InstanceColorGenerator
from nicr_mt_scene_analysis.visualization import PanopticColorGenerator
from nicr_mt_scene_analysis.visualization import visualize_heatmap
from nicr_mt_scene_analysis.visualization import visualize_semantic_pil
from nicr_mt_scene_analysis.visualization import visualize_instance_pil
from nicr_mt_scene_analysis.visualization import visualize_instance
from nicr_mt_scene_analysis.visualization import visualize_orientation
from nicr_mt_scene_analysis.visualization import visualize_instance_orientations
from nicr_mt_scene_analysis.visualization import visualize_instance_center
from nicr_mt_scene_analysis.visualization import visualize_instance_offset
from nicr_mt_scene_analysis.visualization import visualize_panoptic
from nicr_mt_scene_analysis.visualization import visualize_depth
from nicr_scene_analysis_datasets.dataset_base import DatasetConfig
from nicr_scene_analysis_datasets.utils.img import get_visual_distinct_colormap


KWARGS_INSTANCE_ORIENTATION = {
    'thickness': 3,
    'font_size': 45,
    'bg_color': 0,
    'bg_color_font': 'black'
}

KWARGS_INSTANCE_ORIENTATION_WHITEBG = {
    'thickness': 3,
    'font_size': 45,
    'bg_color': 255,
    'bg_color_font': 'white'
}

CV_WRITE_FLAGS = (cv2.IMWRITE_PNG_COMPRESSION, 9)


_shared_color_generators = {
    'instance': None,
    'panoptic': None,
}


def setup_shared_color_generators(dataset_config: DatasetConfig) -> None:
    # instance color generator
    instance_shg = InstanceColorGenerator(
        cmap_without_void=get_visual_distinct_colormap(with_void=False)
    )
    _shared_color_generators['instance'] = instance_shg

    # panoptic color generator
    sem_labels = dataset_config.semantic_label_list
    panoptic_shg = PanopticColorGenerator(
        classes_colors=sem_labels.colors,
        classes_is_thing=sem_labels.classes_is_thing,
        max_instances=(1 << 16),    # we use 16 bit for shifting
        void_label=0
    )
    _shared_color_generators['panoptic'] = panoptic_shg


def visualize(
    output_path: str,
    batch: BatchType,
    predictions: Dict[str, Any],
    dataset_config: DatasetConfig,
    use_shared_color_generators: bool = True,
) -> None:

    # color generators
    if use_shared_color_generators:
        instance_color_generator = _shared_color_generators['instance']
        panoptic_color_generator = _shared_color_generators['panoptic']
        if instance_color_generator is None or panoptic_color_generator is None:
            warnings.warn(
                "Shared color generators are not ready. Please call "
                "'setup_shared_color_generators' first."
            )
    else:
        instance_color_generator = None
        panoptic_color_generator = None

    # visualize ground truth
    gt_path = os.path.join(output_path, 'gt')
    batch_visualization = visualize_batches(
        batch=batch,
        dataset_config=dataset_config,
        instance_color_generator=instance_color_generator,
        panoptic_color_generator=panoptic_color_generator
    )
    save_visualization_result_dict(
        visualization_dict=batch_visualization,
        output_path=gt_path
    )

    # visualize ground truth for side outputs (downscaled images)
    additional_keys = ['_down_8', '_down_16', '_down_32']
    for key in additional_keys:
        if key not in batch:
            # we do not have side outputs
            continue

        # get batch dict for side output and copy identifier
        so_batch = batch[key]
        so_batch['identifier'] = so_batch['identifier']

        # visualize side output
        so_batch_visualization = visualize_batches(
            batch=so_batch,
            dataset_config=dataset_config,
            instance_color_generator=instance_color_generator,
            panoptic_color_generator=panoptic_color_generator
        )
        save_visualization_result_dict(
            visualization_dict=so_batch_visualization,
            output_path=os.path.join(gt_path, key)
        )

    # visualize predictions
    prediction_visualization = visualize_predictions(
        predictions=predictions,
        batch=batch,
        dataset_config=dataset_config,
        instance_color_generator=instance_color_generator,
        panoptic_color_generator=panoptic_color_generator
    )
    save_visualization_result_dict(
        visualization_dict=prediction_visualization,
        output_path=os.path.join(output_path, 'pred')
    )


def save_visualization_result_dict(
    visualization_dict: Dict[str, Any],
    output_path: str
) -> None:
    os.makedirs(output_path, exist_ok=True)
    for key, value in visualization_dict.items():
        if key == 'identifier':
            continue
        for i, v in enumerate(value):
            out_filepath = os.path.join(
                output_path,
                key,
                *visualization_dict['identifier'][i]
            )
            os.makedirs(os.path.dirname(out_filepath), exist_ok=True)

            if isinstance(v, PIL.Image.Image):
                # value is a PIL image
                v.save(out_filepath + '.png')
            elif isinstance(v, np.ndarray):
                # value is an image given as numpy array, write with OpenCV
                if v.ndim == 3:
                    v = cv2.cvtColor(v, cv2.COLOR_RGB2BGR, CV_WRITE_FLAGS)
                cv2.imwrite(out_filepath + '.png', v)
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


def _apply_mask(
    img: np.ndarray,
    mask: np.ndarray,
    value: Union[np.ndarray, Sequence]
) -> None:
    # apply mask inplace
    img[mask, ...] = value
    return img


def _copy_and_apply_mask(
    img: np.ndarray,
    mask: np.ndarray,
    value: Union[np.ndarray, Sequence]
) -> np.ndarray:
    # copy img and apply mask
    return _apply_mask(img.copy(), mask, value)


def visualize_batches(
    batch: BatchType,
    dataset_config: DatasetConfig,
    instance_color_generator: Optional[InstanceColorGenerator] = None,
    panoptic_color_generator: Optional[PanopticColorGenerator] = None,
) -> Dict[str, Any]:
    # note, we use PIL whenever an image with palette is useful

    # semantic colors
    colors = dataset_config.semantic_label_list.colors_array    # with void

    # create dict storing the result
    result_dict = {}
    result_dict['identifier'] = batch['identifier']

    # dump inputs and targets without preprocessing ----------------------------
    if DEFAULT_CLONE_KEY in batch:
        batch_np = batch[DEFAULT_CLONE_KEY]
        # inputs
        if 'rgb' in batch_np:
            result_dict[f'{DEFAULT_CLONE_KEY}_rgb'] = list(batch_np['rgb'])
        if 'depth' in batch_np:
            result_dict[f'{DEFAULT_CLONE_KEY}_depth'] = [
                visualize_depth(img) for img in batch_np['depth']
            ]

        # semantic
        if 'semantic' in batch_np:
            result_dict[f'{DEFAULT_CLONE_KEY}_semantic'] = [
                visualize_semantic_pil(img, colors=colors)
                for img in batch_np['semantic']
            ]

        # instance
        if 'instance' in batch_np:
            result_dict[f'{DEFAULT_CLONE_KEY}_instance'] = [
                visualize_instance_pil(
                    instance_img=img,
                    shared_color_generator=instance_color_generator
                )
                for img in batch_np['instance']
            ]

        # orientation
        if 'orientations' in batch_np:
            result_dict[f'{DEFAULT_CLONE_KEY}_orientations'] = [
                visualize_instance_orientations(
                    *data,
                    shared_color_generator=instance_color_generator,
                    **KWARGS_INSTANCE_ORIENTATION
                ) for data in zip(batch_np['instance'],
                                  batch_np['orientations'])
            ]
            result_dict[f'{DEFAULT_CLONE_KEY}_orientations_white_bg'] = [
                visualize_instance_orientations(
                    *data,
                    shared_color_generator=instance_color_generator,
                    **KWARGS_INSTANCE_ORIENTATION_WHITEBG
                ) for data in zip(batch_np['instance'],
                                  batch_np['orientations'])
            ]

        # scene classification
        if 'scene' in batch_np:
            result_dict[f'{DEFAULT_CLONE_KEY}_scene'] = [
                dataset_config.scene_label_list[s].class_name
                for s in batch_np['scene']
            ]

    else:
        # we do not have the batch data without preprocessing
        batch_np = {}

    # semantic -----------------------------------------------------------------
    if 'semantic' in batch:
        # semantic may have changed due to mapping some classes to void
        result_dict[f'semantic'] = [
            visualize_semantic_pil(img, colors=colors)
            for img in batch['semantic'].cpu().numpy()
        ]

    # instance -----------------------------------------------------------------
    if 'instance' in batch:
        # instance may have changed due to selecting thing classes
        result_dict['instance'] = [
            visualize_instance_pil(
                instance_img=img,
                shared_color_generator=instance_color_generator
            )
            for img in batch['instance'].cpu().numpy()
        ]

        result_dict['instance_white_bg'] = [
            # use foreground mask to change background color to white
            _apply_mask(
                img=visualize_instance(
                    instance_img=img,
                    shared_color_generator=instance_color_generator
                ),
                mask=np.logical_not(fg),
                value=(255, 255, 255)
            )
            for img, fg in zip(batch['instance'].cpu().numpy(),
                               batch['instance_foreground'].cpu().numpy())
        ]

        result_dict['instance_center'] = [
            visualize_instance_center(center_img=img)
            for img in batch['instance_center'].cpu().numpy()
        ]

        result_dict['instance_offset'] = [
            visualize_instance_offset(
                offset_img=img.transpose(1, 2, 0),
                foreground_mask=fg
            )
            for img, fg in zip(batch['instance_offset'].cpu().numpy(),
                               batch['instance_foreground'].cpu().numpy())
        ]

    # orientation --------------------------------------------------------------
    if 'orientation' in batch:
        # instance orientation may have changed due to selecting thing classes
        # 2d dense orientation with black/white background
        result_dict['orientation'] = [
            # use foreground mask to change background color to black
            _apply_mask(
                img=visualize_orientation(o.transpose(1, 2, 0)),
                mask=np.logical_not(fg),
                value=(0, 0, 0)
            )
            for o, fg in zip(batch['orientation'].cpu().numpy(),
                             batch['orientation_foreground'].cpu().numpy())
        ]
        result_dict['orientation_white_bg'] = [
            # change background color to white
            _copy_and_apply_mask(
                img=o_img,
                mask=np.logical_not(fg),
                value=(255, 255, 255)
            )
            for o_img, fg in zip(result_dict['orientation'],
                                 batch['orientation_foreground'].cpu().numpy())
        ]

        # orientation with outline
        result_dict[f'orientations'] = [
            visualize_instance_orientations(
                *data,
                shared_color_generator=instance_color_generator,
                draw_outline=True,
                **KWARGS_INSTANCE_ORIENTATION
            )
            for data in zip(batch['instance'].cpu().numpy(),
                            batch['orientations_present'])
        ]
        result_dict[f'orientations_white_bg'] = [
            visualize_instance_orientations(
                *data,
                shared_color_generator=instance_color_generator,
                draw_outline=True,
                **KWARGS_INSTANCE_ORIENTATION_WHITEBG
            )
            for data in zip(batch['instance'].cpu().numpy(),
                            batch['orientations_present'])
        ]

    # panoptic -----------------------------------------------------------------
    if 'panoptic' in batch:
        sem_labels = dataset_config.semantic_label_list
        result_dict['panoptic'] = [
            visualize_panoptic(
                panoptic_img=img,
                semantic_classes_colors=sem_labels.colors,
                semantic_classes_is_thing=sem_labels.classes_is_thing,
                max_instances=1 << 16,
                void_label=0,
                shared_color_generator=panoptic_color_generator
            )
            for img in batch['panoptic'].cpu().numpy()
        ]

    # panoptic + orientation ---------------------------------------------------
    # panoptic image overlayed with orientation as text
    if 'panoptic' in batch and 'orientations_present' in batch:
        result_dict['panoptic_orientations'] = [
            _copy_and_apply_mask(
                img=panoptic_img,
                mask=visualize_instance_orientations(
                    instance_img=instance,
                    orientations=orientations,
                    shared_color_generator=instance_color_generator,
                    draw_outline=False,
                    thickness=3,
                    font_size=45,
                    bg_color=0,
                    bg_color_font='black'
                ).any(axis=-1),   # text mask
                value=(255, 255, 255)    # white text color
            )
            for panoptic_img, instance, orientations in zip(
                result_dict['panoptic'],
                batch['instance'].cpu().numpy(),
                batch['orientations_present']
            )
        ]

    return result_dict


def visualize_predictions(
    predictions: Dict[str, Any],
    batch: BatchType,
    dataset_config: DatasetConfig,
    instance_color_generator: Optional[InstanceColorGenerator] = None,
    panoptic_color_generator: Optional[PanopticColorGenerator] = None,
) -> Dict[str, Any]:
    # note, we use PIL whenever an image with palette is useful

    # semantic colors and class indices with orientation
    colors = dataset_config.semantic_label_list.colors_array
    use_orientation_class_indices = np.where(
        dataset_config.semantic_label_list.classes_use_orientations
    )[0]

    # create dict for results
    result_dict = {}
    result_dict['identifier'] = batch['identifier']

    # semantic -----------------------------------------------------------------
    # -> predicted class
    key = 'semantic_segmentation_idx'
    if key in predictions:
        for k in (key, get_fullres_key(key)):  # plain output and fullres
            result_dict[k] = [
                visualize_semantic_pil(img, colors=colors[1:])
                for img in predictions[k].cpu().numpy()
            ]
    # -> predicted class score
    key = 'semantic_segmentation_score'
    if key in predictions:
        for k in (key, get_fullres_key(key)):  # plain output and fullres
            result_dict[k] = [
                visualize_heatmap(img, cmap='jet')
                for img in predictions[k].cpu().numpy()
            ]

    # instance -----------------------------------------------------------------
    # -> instance segmentation using gt foreground mask (dataset eval only)
    key = 'instance_segmentation_gt_foreground'
    if key in predictions:
        for k in (key, get_fullres_key(key)):  # plain output and fullres
            result_dict[k] = [
                visualize_instance_pil(
                    instance_img=img,
                    shared_color_generator=instance_color_generator
                )
                for img in predictions[k].cpu().numpy()
            ]

    # raw predictions of instance head (there are no fullres versions)
    # -> instance centers
    key = 'instance_centers'
    if key in predictions:
        result_dict[key] = [
            visualize_instance_center(center_img=img[0])
            for img in predictions[key].cpu().numpy()
        ]
    # -> instance offsets (and masked versions)
    key = 'instance_offsets'
    if key in predictions:
        # plain network output without any mask
        result_dict[key] = [
            visualize_instance_offset(img.transpose(1, 2, 0))
            for img in predictions[key].cpu().numpy()
        ]
        # masked with gt foreground
        key_fg = 'instance_foreground'
        if key_fg in batch:
            result_dict[key+'_gt_foreground'] = [
                _copy_and_apply_mask(
                    img=img_offset,
                    mask=np.logical_not(fg),
                    value=(255, 255, 255)
                ) for img_offset, fg in zip(result_dict[key],
                                            batch[key_fg].cpu().numpy())
            ]
        # masked with predicted foreground for panoptic segmentation
        key_fg = 'panoptic_foreground_mask'
        if key_fg in predictions:
            result_dict[key+'_pred_foreground'] = [
                _copy_and_apply_mask(
                    img=img_offset,
                    mask=np.logical_not(fg),
                    value=(255, 255, 255)
                ) for img_offset, fg in zip(result_dict[key],
                                            predictions[key_fg].cpu().numpy())
            ]

    # (instance) orientation ---------------------------------------------------
    # -> 2d dense raw orientation with black/white background (there is no
    #    fullres version)
    key = 'instance_orientation'
    if key in predictions:
        # plain network output without any mask
        result_dict[key] = [
            visualize_orientation(img.transpose(1, 2, 0))
            for img in predictions[key].cpu().numpy()
        ]
        # masked with gt foreground
        key_fg = 'orientation_foreground'
        if key_fg in batch:
            result_dict[key+'_gt_foreground'] = [
                _copy_and_apply_mask(
                    img=img_o,
                    mask=np.logical_not(fg),
                    value=(0, 0, 0)
                )
                for img_o, fg in zip(
                    result_dict[key],
                    batch['orientation_foreground'].cpu().numpy()
                )
            ]
        # masked with gt foreground and white bg
        key_fg = 'orientation_foreground'
        if key_fg in batch:
            result_dict[key+'_gt_foreground_white_bg'] = [
                _copy_and_apply_mask(
                    img=img_o,
                    mask=np.logical_not(fg),
                    value=(255, 255, 255)
                )
                for img_o, fg in zip(
                    result_dict[key],
                    batch['orientation_foreground'].cpu().numpy()
                )
            ]
        # masked with predicted foreground for panoptic
        key_semantic = 'panoptic_segmentation_deeplab_semantic_idx'
        if key_semantic in predictions:
            fg_masks = [
                np.isin(sem, use_orientation_class_indices)  # both with void
                for sem in predictions[key_semantic].cpu().numpy()
            ]
            # black bg
            result_dict[key+'_pred_foreground'] = [
                _copy_and_apply_mask(
                    img=img_o,
                    mask=np.logical_not(fg),
                    value=(0, 0, 0)
                )
                for img_o, fg in zip(result_dict[key], fg_masks)
            ]
            # white bg
            result_dict[key+'_pred_foreground_white_bg'] = [
                _copy_and_apply_mask(
                    img=img_o,
                    mask=np.logical_not(fg),
                    value=(255, 255, 255)
                )
                for img_o, fg in zip(result_dict[key], fg_masks)
            ]

    # orientations with outline
    # -> predicted orientations with gt instances and gt foreground (dataset
    #    eval only, there is no fullres version)
    key = 'orientations_gt_instance_gt_orientation_foreground'
    if key in predictions:
        result_dict[key] = [
            visualize_instance_orientations(
                *data,
                shared_color_generator=instance_color_generator,
                draw_outline=True,
                **KWARGS_INSTANCE_ORIENTATION
            )
            for data in zip(
                batch['instance'].cpu().numpy(),
                predictions['orientations_gt_instance_gt_orientation_foreground']
            )
        ]
        result_dict[key+'_white_bg'] = [
            visualize_instance_orientations(
                *data,
                shared_color_generator=instance_color_generator,
                draw_outline=True,
                **KWARGS_INSTANCE_ORIENTATION_WHITEBG
            )
            for data in zip(
                batch['instance'].cpu().numpy(),
                predictions['orientations_gt_instance_gt_orientation_foreground']
            )
        ]

    # -> predicted orientations with panoptic instances
    key = 'orientations_panoptic_segmentation_deeplab_instance'
    if key in predictions:
        key_instance = 'panoptic_segmentation_deeplab_instance_idx'
        key_semantic = 'panoptic_segmentation_deeplab_semantic_idx'
        # visualize for both plain output and fullres
        for k_r, k_i, k_s in zip(
            (key, get_fullres_key(key)),
            (key_instance, get_fullres_key(key_instance)),
            (key_semantic, get_fullres_key(key_semantic))
        ):
            # get foreground masks and instance images
            fg_masks = np.isin(predictions[k_s].cpu().numpy(),
                               use_orientation_class_indices)  # both with void
            instance_imgs = predictions[k_i].cpu().numpy()
            instance_imgs[np.logical_not(fg_masks)] = 0

            # black bg
            result_dict[k_r] = [
                visualize_instance_orientations(
                    *data,
                    shared_color_generator=instance_color_generator,
                    draw_outline=True,
                    **KWARGS_INSTANCE_ORIENTATION
                )
                for data in zip(instance_imgs, predictions[key])
            ]
            # white bg
            result_dict[k_r+'_white_bg'] = [
                visualize_instance_orientations(
                    *data,
                    shared_color_generator=instance_color_generator,
                    draw_outline=True,
                    **KWARGS_INSTANCE_ORIENTATION_WHITEBG
                )
                for data in zip(instance_imgs, predictions[key])
            ]

    # panoptic segmentation ----------------------------------------------------
    sem_labels = dataset_config.semantic_label_list

    # -> predicted label
    key = 'panoptic_segmentation_deeplab'
    if key in predictions:
        for k in (key, get_fullres_key(key)):  # plain output and fullres
            result_dict[k] = [
                visualize_panoptic(
                    panoptic_img=img,
                    semantic_classes_colors=sem_labels.colors,
                    semantic_classes_is_thing=sem_labels.classes_is_thing,
                    max_instances=(1 << 16),
                    void_label=0,
                    shared_color_generator=panoptic_color_generator
                )
                for img in predictions[k].cpu().numpy()
            ]

    # -> predicted score
    key = 'panoptic_segmentation_deeplab_panoptic_score'
    if key in predictions:
        for k in (key, get_fullres_key(key)):  # plain output and fullres
            result_dict[k] = [
                visualize_heatmap(img, cmap='jet')
                for img in predictions[k].cpu().numpy()
            ]

    # -> predicted semantic label
    key = 'panoptic_segmentation_deeplab_semantic_idx'
    if key in predictions:
        for k in (key, get_fullres_key(key)):  # plain output and fullres
            result_dict[k] = [
                visualize_semantic_pil(img, sem_labels.colors_array)
                for img in predictions[k].cpu().numpy()
            ]

    # -> predicted semantic score
    key = 'panoptic_segmentation_deeplab_semantic_score'
    if key in predictions:
        for k in (key, get_fullres_key(key)):  # plain output and fullres
            result_dict[k] = [
                visualize_heatmap(img, cmap='jet')
                for img in predictions[k].cpu().numpy()
            ]

    # -> predicted instance label
    key = 'panoptic_segmentation_deeplab_instance_idx'
    if key in predictions:
        for k in (key, get_fullres_key(key)):  # plain output and fullres
            result_dict[k] = [
                visualize_instance_pil(
                    instance_img=img,
                    shared_color_generator=instance_color_generator
                )
                for img in predictions[k].cpu().numpy()
            ]

    # -> predicted instance score
    key = 'panoptic_segmentation_deeplab_instance_score'
    if key in predictions:
        for k in (key, get_fullres_key(key)):  # plain output and fullres
            result_dict[k] = [
                visualize_heatmap(img, cmap='jet')
                for img in predictions[k].cpu().numpy()
            ]

    # everything combined ------------------------------------------------------
    # panoptic segmentation and orientations and rgb overlayed with both
    # (only fullres!)
    if all((
        'panoptic_segmentation_deeplab' in predictions,
        'orientations_panoptic_segmentation_deeplab_instance' in predictions,
    )):
        key_semantic = get_fullres_key('panoptic_segmentation_deeplab_semantic_idx')
        key_instance = get_fullres_key('panoptic_segmentation_deeplab_instance_idx')

        # create orientation images with text but without outline
        fg_masks = np.isin(predictions[key_semantic].cpu().numpy(),
                           use_orientation_class_indices)  # both with void
        instance_imgs = predictions[key_instance].cpu().numpy()
        instance_imgs[np.logical_not(fg_masks)] = 0
        orientation_imgs = [
            visualize_instance_orientations(
                *data,
                shared_color_generator=instance_color_generator,
                draw_outline=False,
                **KWARGS_INSTANCE_ORIENTATION
            )
            for data in zip(
                instance_imgs,
                predictions['orientations_panoptic_segmentation_deeplab_instance']
            )
        ]

        result_dict[get_fullres_key('panoptic_orientations')] = [
            _copy_and_apply_mask(
                img=panoptic_img,
                mask=orientation_img.any(axis=-1),   # text mask
                value=(255, 255, 255)    # white text color
            )
            for panoptic_img, orientation_img in zip(
                result_dict[get_fullres_key('panoptic_segmentation_deeplab')],
                orientation_imgs
            )
        ]

        if DEFAULT_CLONE_KEY in batch:
            result_dict[get_fullres_key('panoptic_orientations_rgb')] = [
                blend_images(
                    img1=panoptic_orientation, img2=rgb, alpha=0.5
                )
                for panoptic_orientation, rgb in zip(
                    result_dict[get_fullres_key('panoptic_orientations')],
                    batch[DEFAULT_CLONE_KEY]['rgb']
                )
            ]

    # scene classification -----------------------------------------------------
    if 'scene_class_idx' in predictions:
        result_dict['scene'] = [
            dataset_config.scene_label_list_without_void[s].class_name
            for s in predictions['scene_class_idx']
        ]

    return result_dict
