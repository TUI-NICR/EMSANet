# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
import torch

from nicr_scene_analysis_datasets import ScanNet


def load_weights(args, model, state_dict, verbose=True):
    # this function accounts for:
    # - renamed keys, e.g., fused_encoders.* -> encoder.*
    # - missing keys, e.g., different number of scene classes
    # - specific dataset or pretraining combinations, e.g., Hypersim -> SUNRGB-D

    print_ = print if verbose else lambda *a, **k: None

    # get current model state dict
    model_state_dict = model.state_dict()

    # the encoder key was renamed from fused_encoders.* to encoder.*
    state_dict = {
        k.replace('fused_encoders.', 'encoder.'): v
        for k, v in state_dict.items()
    }

    if len(state_dict) != len(model_state_dict):
        # loaded state dict is different, run a deeper analysis
        # this can happen if a model trained with deviating tasks is loaded
        # (e.g., pre-training on hypersim with normals)
        # we try to remove the extra keys
        for key in list(state_dict.keys()):
            if key not in model_state_dict:
                print_(f"Removing '{key}' from loaded state dict as the "
                       "current model does not contain such key.")
                _ = state_dict.pop(key)

    # scene classes may differ, e.g., when using pretrained weights on
    # Hypersim for a subsequent training, we skip loading these pretrained
    # weights
    for key in list(state_dict.keys()):
        if all(n in key for n in ('scene_decoder', 'head')):
            n_classes_pretraining = model_state_dict[key].shape[0]
            n_classes_current = state_dict[key].shape[0]
            if n_classes_current != n_classes_pretraining:
                print_(f"Skipping '{key}' as the number of scene classes "
                       f"differs {n_classes_current} (current) vs. "
                       f"{n_classes_pretraining} (pretraining).")
                # we simply use the random weights of the current model
                state_dict[key] = model_state_dict[key]

    if 'semantic' in args.tasks:
        if args.dataset.startswith('nyuv2'):  # first (main) dataset
            # nyuv2 uses 40 semantic classes, when using a checkpoint
            # pretrained on sunrgbd with 37, we can still copy the weights
            # for 37 classes
            for key, weight in list(state_dict.items()):
                if all(n in key for n in ('semantic_decoder', 'head', 'conv')):
                    if weight.shape[0] == 37:
                        weight_sunrgbd = weight.clone()
                        # we simply copy the random weights of the current
                        # model first
                        state_dict[key] = model_state_dict[key]
                        # and then overwrite the first 37 channels
                        print_(f"Reusing 37/40 channels in '{key}'.")
                        state_dict[key][:37, ...] = weight_sunrgbd

        if args.dataset.startswith('sunrgbd'):  # first (main) dataset
            # sunrgbd has only 37 semantic classes, however these classes
            # match the first 37 classes of nyuv2, scannet and hypersim
            # (40 classes), so, if we detect weights with 40 output
            # channels (filter and bias) in a semantic head, we keep the
            # first 37 channels
            for key, weight in list(state_dict.items()):
                if all(n in key for n in ('semantic_decoder', 'head', 'conv')):
                    if weight.shape[0] == 40:
                        print_(f"Removing last 3 channels in '{key}'.")
                        state_dict[key] = weight[:37, ...]

        elif args.dataset.startswith('scannet'):  # first (main) dataset
            # check if training (e.g., pretraining on hypersim) was done
            # with more classes, we can handle two cases 40 -> 20 and
            # 549 -> 200
            if not args.validation_scannet_benchmark_mode:
                # otherwise, we already would have 20 / 200 classes

                # get mapping and mask
                if 20 == args.scannet_semantic_n_classes:
                    mapping = ScanNet.SEMANTIC_CLASSES_40_MAPPING_TO_BENCHMARK
                else:
                    mapping = ScanNet.SEMANTIC_CLASSES_549_MAPPING_TO_BENCHMARK200

                mask = torch.tensor([
                    c_benchmark != 0   # class is not ignored
                    for c_data, c_benchmark in mapping.items()
                    if c_data != 0     # skip void class
                ], dtype=torch.bool)

                # check weights of semantic heads and remove ignored classes
                for key, weight in list(state_dict.items()):
                    if all(n in key for n in ('semantic_decoder', 'head',
                                              'conv')):
                        if weight.shape[0] == mask.shape[0]:
                            print_("Removing channels for ignored classes "
                                   f"in '{key}'.")
                            state_dict[key] = weight[mask, ...]

        # remove all semantic weights if shape still does not match,
        # happens, e.g., when using pretrained weights from scannet with
        # 20 classes for sunrbgd or nyuv2
        for key, weight in list(state_dict.items()):
            if all(n in key for n in ('semantic_decoder', 'head', 'conv')):
                if weight.shape != model_state_dict[key].shape:
                    print_(f"Removing '{key}' from loaded state dict as"
                           f"the shape does not match: {weight.shape} "
                           f"vs. {model_state_dict[key].shape}.")
                    # we simply use the random weights of the current
                    # model
                    state_dict[key] = model_state_dict[key]

    model.load_state_dict(state_dict, strict=True)
