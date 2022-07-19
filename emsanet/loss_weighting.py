# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from nicr_mt_scene_analysis.loss_weighting import FixedLossWeighting
from nicr_mt_scene_analysis.loss_weighting import LossWeightingType

from nicr_mt_scene_analysis.task_helper.base import get_total_loss_key


def get_loss_weighting_module(args) -> LossWeightingType:
    # we stick to fixed task weighting as none of the remaining was working well

    # assign weight to each task (based on positional order)
    tasks_weights = {}
    assert len(args.tasks) == len(args.tasks_weighting)
    tasks_weights = {
        task: weight
        for task, weight in zip(args.tasks, args.tasks_weighting)
    }

    # convert task weights to loss weights (keys must match the later losses)
    # note, we consider only losses marked as total for weighting for now
    loss_weights = {}

    # handle orientation as it is part of the instance decoder
    if 'orientation' in args.tasks:
        loss_weights[get_total_loss_key('instance_orientation')] = \
            tasks_weights.pop('orientation')

    # handle instance keys
    if 'instance' in args.tasks:
        # overall weight for the instance task (not orientation!)
        weight_instance = tasks_weights.pop('instance')
        # additional weighting for both instance tasks (center and offset)
        weight_center, weight_offset = args.instance_weighting
        # to determine the final (flat) weights, values gets multiplied
        loss_weights[get_total_loss_key('instance_center')] = \
            weight_instance*weight_center
        loss_weights[get_total_loss_key('instance_offset')] = \
            weight_instance*weight_offset

    # for the remaining tasks, simply append the total loss suffix
    loss_weights.update({
        get_total_loss_key(task): value
        for task, value in tasks_weights.items()
    })

    return FixedLossWeighting(weights=loss_weights)
