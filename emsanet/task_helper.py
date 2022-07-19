# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Tuple

from nicr_mt_scene_analysis.task_helper import NormalTaskHelper
from nicr_mt_scene_analysis.task_helper import SemanticTaskHelper
from nicr_mt_scene_analysis.task_helper import SceneTaskHelper
from nicr_mt_scene_analysis.task_helper import InstanceTaskHelper
from nicr_mt_scene_analysis.task_helper import PanopticTaskHelper
from nicr_mt_scene_analysis.task_helper import TaskHelperType

from .data import DatasetType


def get_task_helpers(
    args,
    dataset: DatasetType
) -> Tuple[TaskHelperType]:
    task_helper = []

    if 'semantic' in args.tasks:
        class_weights = dataset.semantic_compute_class_weights(
            weight_mode=args.semantic_class_weighting,
            c=args.semantic_class_weighting_logarithmic_c,
            n_threads=4,
            debug=False
        )
        task_helper.append(
            SemanticTaskHelper(
                n_classes=dataset.semantic_n_classes_without_void,
                class_weights=class_weights,
                label_smoothing=args.semantic_loss_label_smoothing,
                disable_multiscale_supervision=args.semantic_no_multiscale_supervision,
                examples_cmap=dataset.semantic_class_colors_without_void
            )
        )
    if 'scene' in args.tasks:
        task_helper.append(
            SceneTaskHelper(
                n_classes=dataset.scene_n_classes_without_void,
                class_weights=None,
                label_smoothing=args.scene_loss_label_smoothing
            )
        )
    if 'normal' in args.tasks:
        task_helper.append(
            NormalTaskHelper(
                loss_name=args.normal_loss,
                disable_multiscale_supervision=args.normal_no_multiscale_supervision
            )
        )
    if 'instance' in args.tasks or 'orientation' in args.tasks:
        task_helper.append(
            InstanceTaskHelper(
                semantic_n_classes=dataset.semantic_n_classes,
                semantic_classes_is_thing=dataset.config.semantic_label_list.classes_is_thing,
                loss_name_instance_center=args.instance_center_loss,
                disable_multiscale_supervision=args.instance_no_multiscale_supervision
            )
        )
    if args.enable_panoptic:
        task_helper.append(
            PanopticTaskHelper(
                semantic_n_classes=dataset.semantic_n_classes,
                semantic_classes_is_thing=dataset.config.semantic_label_list.classes_is_thing,
                semantic_label_list=dataset.config.semantic_label_list,
            )
        )
    return tuple(task_helper)
