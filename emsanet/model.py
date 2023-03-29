# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Any, Dict

from collections import ChainMap

from nicr_mt_scene_analysis.model.block import get_block_class
from nicr_mt_scene_analysis.model.backbone import get_backbone
from nicr_mt_scene_analysis.model.context_module import get_context_module

from nicr_mt_scene_analysis.model.encoder import get_encoder
from nicr_mt_scene_analysis.model.encoder_fusion import get_encoder_fusion_class
from nicr_mt_scene_analysis.model.encoder_decoder_fusion import get_encoder_decoder_fusion_class
from nicr_mt_scene_analysis.model.upsampling import Upsampling
from nicr_mt_scene_analysis.model.initialization import he_initialization
from nicr_mt_scene_analysis.model.initialization import zero_residual_initialization
import torch

from .data import DatasetConfig
from .decoder import get_decoders


class EMSANet(torch.nn.Module):
    def __init__(
        self,
        args,
        dataset_config: DatasetConfig
    ) -> None:
        super().__init__()

        # store args and dataset parameters
        self.args = args
        self.dataset_config = dataset_config

        # get some dataset properties
        semantic_labels = dataset_config.semantic_label_list_without_void
        semantic_n_classes = len(semantic_labels)
        scene_n_classes = len(dataset_config.scene_label_list_without_void)
        panoptic_semantic_classes_is_thing = semantic_labels.classes_is_thing
        panoptic_use_orientation = tuple(semantic_labels.classes_use_orientations)

        # create encoder(s)
        if 'rgb' in args.input_modalities:
            backbone_rgb = get_backbone(
                name=args.rgb_encoder_backbone,
                resnet_block=get_block_class(
                    args.rgb_encoder_backbone_resnet_block,
                    dropout_p=args.dropout_p
                ),
                n_input_channels=3,
                normalization=args.encoder_normalization,
                activation=args.activation,
                pretrained=not args.no_pretrained_backbone,
                pretrained_filepath=args.rgb_encoder_backbone_pretrained_weights_filepath
            )
        else:
            backbone_rgb = None

        if 'depth' in args.input_modalities:
            backbone_depth = get_backbone(
                name=args.depth_encoder_backbone,
                resnet_block=get_block_class(
                    args.depth_encoder_backbone_resnet_block,
                    dropout_p=args.dropout_p
                ),
                n_input_channels=1,
                normalization=args.encoder_normalization,
                activation=args.activation,
                pretrained=not args.no_pretrained_backbone,
                pretrained_filepath=args.depth_encoder_backbone_pretrained_weights_filepath
            )
        else:
            backbone_depth = None

        if 'rgbd' in args.input_modalities:
            backbone_rgbd = get_backbone(
                name=args.rgbd_encoder_backbone,
                resnet_block=get_block_class(
                    args.rgbd_encoder_backbone_resnet_block,
                    dropout_p=args.dropout_p
                ),
                n_input_channels=3+1,
                normalization=args.encoder_normalization,
                activation=args.activation,
                pretrained=not args.no_pretrained_backbone,
                pretrained_filepath=args.rgbd_encoder_backbone_pretrained_weights_filepath
            )
        else:
            backbone_rgbd = None

        # fuse encoder(s) in a shared module
        self.encoder = get_encoder(
            backbone_rgb=backbone_rgb,
            backbone_depth=backbone_depth,
            backbone_rgbd=backbone_rgbd,
            fusion=args.encoder_fusion,
            normalization=args.encoder_normalization,
            activation=args.activation,
            skip_downsamplings=args.encoder_decoder_skip_downsamplings
        )
        enc_downsampling = self.encoder.downsampling
        enc_n_channels_out = self.encoder.n_channels_out
        enc_skips_n_channels = self.encoder.skips_n_channels

        # create context module
        self.context_module = get_context_module(
            name=args.context_module,
            n_channels_in=enc_n_channels_out,
            n_channels_out=enc_n_channels_out,
            input_size=(args.input_height // enc_downsampling,
                        args.input_width // enc_downsampling),
            # context module only makes sense with batch normalization
            normalization='bn',
            activation=args.activation,
            upsampling=args.upsampling_context_module
        )

        # create decoder(s)
        if args.instance_offset_encoding == 'tanh':
            instance_normalized_offset = True
            instance_tanh_for_offset = True
        elif args.instance_offset_encoding == 'relative':
            instance_normalized_offset = True
            instance_tanh_for_offset = False
        elif args.instance_offset_encoding == 'deeplab':
            instance_normalized_offset = False
            instance_tanh_for_offset = False
        else:
            raise NotImplementedError

        if args.instance_center_encoding == 'sigmoid':
            instance_sigmoid_for_center = True
        else:
            instance_sigmoid_for_center = False

        self.decoders = get_decoders(
            args,
            n_channels_in=enc_n_channels_out,
            downsampling_in=enc_downsampling,
            # semantic segmentation
            semantic_n_classes=semantic_n_classes,
            # instance segmentation
            instance_normalized_offset=instance_normalized_offset,
            instance_offset_distance_threshold=args.instance_offset_distance_threshold,
            instance_sigmoid_for_center=instance_sigmoid_for_center,
            instance_tanh_for_offset=instance_tanh_for_offset,
            # surface normal estimation
            normal_n_channels_out=3,
            # scene classification
            scene_n_channels_in=self.context_module.n_channels_reduction,
            scene_n_classes=scene_n_classes,
            # panoptic
            panoptic_semantic_classes_is_thing=panoptic_semantic_classes_is_thing,
            panoptic_has_orientation=panoptic_use_orientation,
            # other shared args
            fusion_n_channels=enc_skips_n_channels[::-1],
        )

        # initialization
        debug_init = args.debug
        # apply he initialization to selected parts of the network
        for part in args.he_init:
            # whitelisted initialization
            cls = None
            if 'encoder-fusion' == part:
                cls = get_encoder_fusion_class(args.encoder_fusion)
            elif 'encoder-decoder-fusion' == part:
                cls = get_encoder_decoder_fusion_class(
                    args.encoder_decoder_fusion
                )

            if cls is not None:
                for n, m in self.named_modules():
                    if isinstance(m, cls):
                        he_initialization(m, name_hint=n, debug=debug_init)

            # (blacklisted) initialization
            if 'context-module' == part:
                he_initialization(self.context_module, debug=debug_init)
            elif 'decoder' == part:
                he_initialization(self.decoders,
                                  blacklist=(Upsampling,),
                                  debug=debug_init)

        # init last norm in residuals to zero to enforce identity on start
        if not args.no_zero_init_decoder_residuals:
            zero_residual_initialization(self.decoders, debug=debug_init)

    def forward(self, batch, do_postprocessing=False) -> Dict[str, Any]:
        # determine input
        enc_inputs = {}
        if 'rgbd' in self.args.input_modalities:
            rgb = batch['rgb']
            depth = batch['depth']
            rgbd = torch.cat([rgb, depth], dim=1)
            enc_inputs['rgbd'] = rgbd
        else:
            if 'rgb' in self.args.input_modalities:
                enc_inputs['rgb'] = batch['rgb']
            if 'depth' in self.args.input_modalities:
                enc_inputs['depth'] = batch['depth']
        # forward (fused) encoder(s)
        enc_outputs, enc_dec_skips = self.encoder(enc_inputs)

        # forward context module
        if len(self.args.input_modalities) == 2:
            # design choice up to now, use output of rgb encoder as input for
            # context module
            con_input = enc_outputs['rgb']
        else:
            # use the output of the decoder with the same name as the input
            assert len(enc_inputs) == 1    # only one input modality
            con_input = enc_outputs[list(enc_inputs.keys())[0]]
        con_outputs, con_context_outputs = self.context_module(con_input)

        # forward decoder(s)
        outputs = []
        for decoder in self.decoders.values():
            outputs.append(
                decoder(
                    (con_outputs, con_context_outputs), enc_dec_skips, batch,
                    do_postprocessing=do_postprocessing
                )
            )

        # simplify output if postprocessing was applied
        if do_postprocessing:
            outputs = dict(ChainMap(*outputs))

        return outputs
