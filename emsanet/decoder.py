# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Tuple, Union

from torch import nn

from nicr_mt_scene_analysis.model.activation import get_activation_class
from nicr_mt_scene_analysis.model.block import get_block_class
from nicr_mt_scene_analysis.model.decoder import SemanticDecoder
from nicr_mt_scene_analysis.model.decoder import SemanticMLPDecoder
from nicr_mt_scene_analysis.model.decoder import InstanceDecoder
from nicr_mt_scene_analysis.model.decoder import InstanceMLPDecoder
from nicr_mt_scene_analysis.model.decoder import NormalDecoder
from nicr_mt_scene_analysis.model.decoder import NormalMLPDecoder
from nicr_mt_scene_analysis.model.decoder import PanopticHelper
from nicr_mt_scene_analysis.model.decoder import SceneClassificationDecoder
from nicr_mt_scene_analysis.model.encoder_decoder_fusion import get_encoder_decoder_fusion_class
from nicr_mt_scene_analysis.model.normalization import get_normalization_class
from nicr_mt_scene_analysis.model.postprocessing import get_postprocessing_class
from nicr_mt_scene_analysis.model.upsampling import get_upsampling_class


KNOWN_DECODERS = (
    'emsanet',    # decoder used in EMSANet publication
    'segformermlp',    # MLP decoder used in SegFormer publication
)


def get_decoders(
    args,
    n_channels_in: int,
    downsampling_in: int,
    semantic_n_classes: int = 40,
    instance_normalized_offset: bool = True,
    instance_offset_distance_threshold: Union[None, int] = None,
    instance_sigmoid_for_center: bool = True,
    instance_tanh_for_offset: bool = True,
    panoptic_semantic_classes_is_thing: Tuple[bool, ...] = (True, )*40,
    panoptic_has_orientation: Tuple[bool, ...] = (True, )*40,
    normal_n_channels_out: int = 3,
    scene_n_channels_in: int = 512//2,
    scene_n_classes: int = 10,
    fusion_n_channels: Tuple[int, ...] = (512, 256, 128),
    **kwargs
) -> nn.ModuleList:

    # common parameters used in almost all encoders
    common_kwargs = {
        'n_channels_in': n_channels_in,
        'downsampling_in': downsampling_in,
        'fusion_n_channels': fusion_n_channels,
        'fusion_downsamplings': args.encoder_decoder_skip_downsamplings[::-1],
        'normalization': get_normalization_class(args.decoder_normalization),
        'activation': get_activation_class(args.activation),
        'prediction_upsampling': get_upsampling_class(args.upsampling_prediction)
    }

    decoders = {}
    # semantic segmentation
    if 'semantic' in args.tasks:
        decoder_type = args.semantic_decoder.lower()
        if 'emsanet' == decoder_type:
            semantic_decoder = SemanticDecoder(
                n_channels=args.semantic_decoder_n_channels,
                downsamplings=args.semantic_decoder_downsamplings,
                block=get_block_class(
                    args.semantic_decoder_block,
                    dropout_p=args.semantic_decoder_block_dropout_p
                ),
                n_blocks=args.semantic_decoder_n_blocks,
                fusion=get_encoder_decoder_fusion_class(args.semantic_encoder_decoder_fusion),
                n_classes=semantic_n_classes,
                postprocessing=get_postprocessing_class('semantic', **kwargs),
                upsampling=get_upsampling_class(args.semantic_decoder_upsampling),
                **common_kwargs
            )
        elif 'segformermlp' == decoder_type:
            semantic_decoder = SemanticMLPDecoder(
                n_channels=args.semantic_decoder_n_channels,
                fusion=get_encoder_decoder_fusion_class(args.semantic_encoder_decoder_fusion),
                n_classes=semantic_n_classes,
                downsampling_in_heads=4,
                dropout_p=args.semantic_decoder_dropout_p,
                postprocessing=get_postprocessing_class('semantic', **kwargs),
                upsampling=get_upsampling_class(args.semantic_decoder_upsampling),
                **common_kwargs
            )
        decoders['semantic_decoder'] = semantic_decoder

    # (class-agnostic) instance segmentation
    if 'instance' in args.tasks:
        instance_postprocessing = get_postprocessing_class(
            'instance',
            heatmap_threshold=args.instance_center_heatmap_threshold,
            heatmap_nms_kernel_size=args.instance_center_heatmap_nms_kernel_size,
            heatmap_apply_foreground_mask=args.instance_center_heatmap_apply_foreground_mask,
            top_k_instances=args.instance_center_heatmap_top_k,
            normalized_offset=instance_normalized_offset,
            offset_distance_threshold=instance_offset_distance_threshold,
            **kwargs
        )

        decoder_type = args.instance_decoder.lower()
        if 'emsanet' == decoder_type:
            instance_decoder = InstanceDecoder(
                n_channels=args.instance_decoder_n_channels,
                downsamplings=args.instance_decoder_downsamplings,
                block=get_block_class(
                    args.instance_decoder_block,
                    dropout_p=args.instance_decoder_block_dropout_p
                ),
                n_blocks=args.instance_decoder_n_blocks,
                fusion=get_encoder_decoder_fusion_class(args.instance_encoder_decoder_fusion),
                n_channels_per_task=32,    # default panoptic deeplab
                with_orientation=('orientation' in args.tasks),
                sigmoid_for_center=instance_sigmoid_for_center,
                tanh_for_offset=instance_tanh_for_offset,
                postprocessing=instance_postprocessing,
                upsampling=get_upsampling_class(args.instance_decoder_upsampling),
                **common_kwargs
            )
        elif 'segformermlp' == decoder_type:
            instance_decoder = InstanceMLPDecoder(
                n_channels=args.instance_decoder_n_channels,
                fusion=get_encoder_decoder_fusion_class(args.instance_encoder_decoder_fusion),
                n_channels_per_task=32,    # default panoptic deeplab
                with_orientation=('orientation' in args.tasks),
                sigmoid_for_center=instance_sigmoid_for_center,
                tanh_for_offset=instance_tanh_for_offset,
                downsampling_in_heads=4,
                dropout_p=args.instance_decoder_dropout_p,
                postprocessing=instance_postprocessing,
                upsampling=get_upsampling_class(args.instance_decoder_upsampling),
                **common_kwargs
            )
        decoders['instance_decoder'] = instance_decoder

    # panoptic segmentation
    if args.enable_panoptic:
        panoptic_helper = PanopticHelper(
            semantic_decoder=semantic_decoder,
            instance_decoder=instance_decoder,
            postprocessing=get_postprocessing_class(
                'panoptic',
                semantic_postprocessing=semantic_decoder.postprocessing,
                instance_postprocessing=instance_decoder.postprocessing,
                semantic_classes_is_thing=panoptic_semantic_classes_is_thing,
                semantic_class_has_orientation=panoptic_has_orientation,
                compute_scores=True,
                **kwargs
            )
        )
        # replace dict with decoders (can only contain semantic and instance
        # up to now)
        decoders = {'panoptic_helper': panoptic_helper}

    # surface normal estimation
    if 'normal' in args.tasks:
        decoder_type = args.normal_decoder.lower()
        if 'emsanet' == decoder_type:
            normal_decoder = NormalDecoder(
                n_channels=args.normal_decoder_n_channels,
                downsamplings=args.normal_decoder_downsamplings,
                block=get_block_class(
                    args.normal_decoder_block,
                    dropout_p=args.normal_decoder_block_dropout_p
                ),
                n_blocks=args.normal_decoder_n_blocks,
                fusion=get_encoder_decoder_fusion_class(args.normal_encoder_decoder_fusion),
                n_channels_out=normal_n_channels_out,
                postprocessing=get_postprocessing_class('normal', **kwargs),
                upsampling=get_upsampling_class(args.normal_decoder_upsampling),
                **common_kwargs
            )
        elif 'segformermlp' == decoder_type:
            normal_decoder = NormalMLPDecoder(
                n_channels=args.normal_decoder_n_channels,
                fusion=get_encoder_decoder_fusion_class(args.normal_encoder_decoder_fusion),
                n_channels_out=normal_n_channels_out,
                downsampling_in_heads=4,
                dropout_p=args.normal_decoder_dropout_p,
                postprocessing=get_postprocessing_class('normal', **kwargs),
                upsampling=get_upsampling_class(args.normal_decoder_upsampling),
                **common_kwargs
            )
        decoders['normal_decoder'] = normal_decoder

    # scene classification
    if 'scene' in args.tasks:
        common_kwargs['n_channels_in'] = scene_n_channels_in
        scene_decoder = SceneClassificationDecoder(
            n_classes=scene_n_classes,
            postprocessing=get_postprocessing_class('scene', **kwargs),
            **common_kwargs
        )
        decoders['scene_decoder'] = scene_decoder

    return nn.ModuleDict(decoders)
