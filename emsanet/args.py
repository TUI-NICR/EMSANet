# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
"""
import argparse as ap
import json
import os
import shlex
import shutil
import socket

from nicr_mt_scene_analysis.model.activation import KNOWN_ACTIVATIONS
from nicr_mt_scene_analysis.model.backbone import KNOWN_BACKBONES
from nicr_mt_scene_analysis.model.block import KNOWN_BLOCKS
from nicr_mt_scene_analysis.model.context_module import KNOWN_CONTEXT_MODULES
from nicr_mt_scene_analysis.model.encoder_decoder_fusion import KNOWN_ENCODER_DECODER_FUSIONS
from nicr_mt_scene_analysis.model.encoder_fusion import KNOWN_ENCODER_FUSIONS
from nicr_mt_scene_analysis.model.normalization import KNOWN_NORMALIZATIONS
from nicr_mt_scene_analysis.model.upsampling import KNOWN_UPSAMPLING_METHODS
from nicr_mt_scene_analysis.multi_task import KNOWN_TASKS
from nicr_mt_scene_analysis.task_helper.instance import KNOWN_INSTANCE_CENTER_LOSS_FUNCTIONS
from nicr_mt_scene_analysis.task_helper.normal import KNOWN_NORMAL_LOSS_FUNCTIONS

from .data import KNOWN_DATASETS
from .data import KNOWN_CLASS_WEIGHTINGS
from .decoder import KNOWN_DECODERS
from .lr_scheduler import KNOWN_LR_SCHEDULERS
from .optimizer import KNOWN_OPTIMIZERS


class Range(object):
    """
    Helper for argparse to restrict floats to be in a specified range.
    """
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end

    def __contains__(self, item):
        return self.__eq__(item)

    def __iter__(self):
        yield self

    def __repr__(self):
        return f'[{self.start}, {self.end}]'


class ArgParserEMSANet(ap.ArgumentParser):
    def __init__(self, *args, **kwargs):
        # force ArgumentDefaultsHelpFormatter as formatter_class is given
        formatter_class = kwargs.pop('formatter_class', None)
        formatter_class = formatter_class or ap.ArgumentDefaultsHelpFormatter

        super().__init__(*args, formatter_class=formatter_class, **kwargs)

        # paths ---------------------------------------------------------------
        group = self.add_argument_group('Paths')
        group.add_argument(
             '--results-basepath',
             type=str,
             default='./results',
             help="Path where to store training files."
        )
        group.add_argument(
            '--weights-filepath',
            type=str,
            default=None,
            help="Filepath to (last) checkpoint / weights for the entire model."
        )

        # network and multi-task -----------------------------------------------
        group = self.add_argument_group('Tasks')
        # -> multi-task parameters
        group.add_argument(
            '--tasks',
            nargs='+',
            type=str,
            choices=KNOWN_TASKS,
            default=('semantic',),
            help="Task(s) to perform."
        )
        group.add_argument(
            '--enable-panoptic',
            action='store_true',
            default=False,
            help="Enforces taskts 'semanic' and 'instance' to be combined for "
                 "panoptic segmentation"
        )

        # -> input
        group = self.add_argument_group('Input')
        group.add_argument(
            '--input-height',
            type=int,
            default=480,
            help="Network input height. Images will be resized to this height."
        )
        group.add_argument(
            '--input-width',
            type=int,
            default=640,
            help="Network input width. Images will be resized to this width."
        )
        group.add_argument(
            '--input-modalities',
            nargs='+',
            type=str,
            choices=('rgb', 'depth', 'rgbd'),
            default=('rgb', 'depth'),
            help="Input modalities to consider."
        )

        # -> whole model
        group = self.add_argument_group('Model')
        group.add_argument(
            '--normalization',
            type=str,
            default=None,
            choices=KNOWN_NORMALIZATIONS,
            help="[DEPRECATED - use encoder or decoder specific] Normalization "
                 "to apply in the whole model."
        )
        group.add_argument(
            '--activation',
            type=str,
            default='relu',
            choices=KNOWN_ACTIVATIONS,
            help="Activation to use in the whole model."
        )

        # -> encoder related parameters
        group = self.add_argument_group('Model: Encoder(s)')
        group.add_argument(
            '--no-pretrained-backbone',
            action='store_true',
            default=False,
            help="Disables loading of ImageNet pretrained weights for the "
                 "backbone(s). Useful for inference or inference timing."
        )
        group.add_argument(
            '--encoder-normalization',
            type=str,
            default='batchnorm',
            choices=KNOWN_NORMALIZATIONS,
            help="Normalization to apply to the encoders."
        )
        group.add_argument(
            '--encoder-backbone-pretrained-weights-filepath',
            type=str,
            default=None,
            help="Path to pretrained (ImageNet) weights for the encoder "
                 "backbones. Use this argument if you want to initialize all "
                 "encoder backbones with the same weights. "
                 "If `weights-filepath` is given, the specified weights are "
                 "loaded subsequently and may replace the pretrained weights."
        )
        group.add_argument(
            '--encoder-fusion',
            choices=KNOWN_ENCODER_FUSIONS,
            default='se-add-uni-rgb',
            help="Determines how features of the depth (rgb) encoder are "
                 "fused to features of the other encoder."
        )
        # -> rgb encoder
        group = self.add_argument_group('Model: Encoder(s) -> RGB encoder')
        group.add_argument(
            '--rgb-encoder-backbone',
            type=str,
            choices=KNOWN_BACKBONES,
            default='resnet34',
            help="Backbone to use for RGB encoder."
        )
        group.add_argument(
            '--rgb-encoder-backbone-resnet-block',
            type=str,
            choices=KNOWN_BLOCKS,
            default='nonbottleneck1d',
            help="Block (type) to use in RGB encoder backbone."
        )
        group.add_argument(
            '--rgb-encoder-backbone-block',
            type=str,
            choices=KNOWN_BLOCKS,
            default=None,
            help="[DEPRECATED - use rgb-encoder-backbone-resnet-block] "
                 "Block (type) to use in RGB encoder backbone."
        )
        group.add_argument(
            '--rgb-encoder-backbone-pretrained-weights-filepath',
            type=str,
            default=None,
            help="Path to pretrained (ImageNet) weights for the rgb encoder "
                 "backbone. "
                 "If `weights-filepath` is given, the specified weights are "
                 "loaded subsequently and may replace the pretrained weights."
        )
        # -> depth encoder
        group = self.add_argument_group('Model: Encoder(s) -> depth encoder')
        group.add_argument(
            '--depth-encoder-backbone',
            type=str,
            choices=KNOWN_BACKBONES,
            default='resnet34',
            help="Backbone to use for depth encoder."
        )
        group.add_argument(
            '--depth-encoder-backbone-resnet-block',
            type=str,
            choices=KNOWN_BLOCKS,
            default='nonbottleneck1d',
            help="Block (type) to use in depth encoder backbone."
        )
        group.add_argument(
            '--depth-encoder-backbone-block',
            type=str,
            choices=KNOWN_BLOCKS,
            default=None,
            help="[DEPRECATED - use depth-encoder-backbone-resnet-block] "
                 "Block (type) to use in depth encoder backbone."
        )
        group.add_argument(
            '--depth-encoder-backbone-pretrained-weights-filepath',
            type=str,
            default=None,
            help="Path to pretrained (ImageNet) weights for the depth encoder "
                 "backbone. "
                 "If `weights-filepath` is given, the specified weights are "
                 "loaded subsequently and may replace the pretrained weights."
        )
        # -> rgbd encoder
        group = self.add_argument_group('Model: Encoder(s) -> RGB-D encoder')
        group.add_argument(
            '--rgbd-encoder-backbone',
            type=str,
            choices=KNOWN_BACKBONES,
            default='resnet34',
            help="Backbone to use for RGBD encoder."
        )
        group.add_argument(
            '--rgbd-encoder-backbone-resnet-block',
            type=str,
            choices=KNOWN_BLOCKS,
            default='nonbottleneck1d',
            help="Block (type) to use in RGBD encoder backbone."
        )
        group.add_argument(
            '--rgbd-encoder-backbone-pretrained-weights-filepath',
            type=str,
            default=None,
            help="Path to pretrained (ImageNet) weights for the rgbd encoder "
                 "backbone. "
                 "If `weights-filepath` is given, the specified weights are "
                 "loaded subsequently and may replace the pretrained weights."
        )

        # -> context module related parameters
        group = self.add_argument_group('Model: Context Module')
        group.add_argument(
            '--context-module',
            type=str,
            choices=KNOWN_CONTEXT_MODULES,
            default='ppm',
            help='Context module to use.'
        )
        group.add_argument(
            '--upsampling-context-module',
            choices=('nearest', 'bilinear'),
            default='bilinear',
            help="How features are upsampled in the context module. Bilinear "
                 "upsampling may cause problems when converting to TensorRT."
        )

        # -> decoder related parameters
        group = self.add_argument_group('Model: Decoder(s)')
        group.add_argument(
            '--encoder-decoder-skip-downsamplings',
            nargs='+',
            type=int,
            default=(4, 8, 16),
            help="Determines at which downsamplings skip connections from the "
                 "encoder to the decoder(s) should be created, e.g., '4, 8' "
                 "means skip connections after encoder stages at 1/4 and 1/8 "
                 "of the input size to the decoder."
        )
        group.add_argument(
            '--encoder-decoder-fusion',
            type=str,
            choices=KNOWN_ENCODER_DECODER_FUSIONS,
            default=None,
            help="[DEPRECATED - use parameter for each decoder] Determines "
                 "how features of the encoder (after fusing "
                 "encoder features) are fused into the decoders."
        )
        group.add_argument(
            '--upsampling-decoder',
            choices=KNOWN_UPSAMPLING_METHODS,
            default=None,
            help="[DEPRECATED - use parameter for each decoder] How features "
                 "are upsampled in the decoders. Bilinear upsampling may "
                 "cause problems when converting to TensorRT. 'learned-3x3*' "
                 "mimics bilinear interpolation with nearest interpolation "
                 "and adaptable 3x3 depth-wise convolution subsequently."
        )
        group.add_argument(
            '--upsampling-prediction',
            choices=KNOWN_UPSAMPLING_METHODS,
            default='learned-3x3-zeropad',
            help="How features are upsampled after the last decoder module to "
                 "match the NETWORK input resolution. Bilinear upsampling may "
                 "cause problems when converting to TensorRT. 'learned-3x3*' "
                 "mimics bilinear interpolation with nearest interpolation "
                 "and adaptable 3x3 depth-wise conv subsequently."
        )
        group.add_argument(
            '--decoder-normalization',
            type=str,
            default='batchnorm',
            choices=KNOWN_NORMALIZATIONS,
            help="Normalization to apply in the decoder."
        )

        # -> semantic related parameters
        group = self.add_argument_group('Model: Decoder(s) -> Semantic')
        group.add_argument(
            '--semantic-encoder-decoder-fusion',
            type=str,
            choices=KNOWN_ENCODER_DECODER_FUSIONS,
            default='add-rgb',
            help="Determines how features of the encoder (after fusing "
                 "encoder features) are fused into the semantic decoder."
        )
        group.add_argument(
            '--semantic-decoder',
            type=str,
            default='emsanet',
            choices=KNOWN_DECODERS,
            help="Decoder type to use for semantic segmentation."
        )
        group.add_argument(
            '--semantic-decoder-block',
            type=str,
            default='nonbottleneck1d',
            choices=KNOWN_BLOCKS,
            help="[EMSANet decoder] Block (type) to use in semantic decoder."
        )
        group.add_argument(
            '--semantic-decoder-block-dropout-p',
            type=float,
            default=0.2,
            help="[EMSANet decoder] Dropout probability to use in semantic "
                 "decoder blocks (only for 'nonbottleneck1d')."
        )
        group.add_argument(
            '--semantic-decoder-n-blocks',
            type=int,
            default=3,
            help="[EMSANet decoder] Number of blocks to use in each semantic "
                 "decoder module."
        )
        group.add_argument(
            '--semantic-decoder-dropout-p',
            type=float,
            default=0.1,
            help="[SegFormerMLP decoder] Probability to use for feature "
                 "dropout (Dropout2d) in semantic decoder before task head."
        )
        group.add_argument(
            '--semantic-decoder-n-channels',
            type=int,
            default=(512, 256, 128),
            nargs='+',
            help="[EMSANet decoder] Number of features maps (channels) to use "
                 "in each semantic decoder module. Length of tuple "
                 "determines the number of decoder modules. "
                 "[SegFormerMLP decoder] Embedding dimensions to use for main "
                 "branch and skip connections."
        )
        group.add_argument(
            '--semantic-decoder-downsamplings',
            type=int,
            default=(16, 8, 4),
            nargs='+',
            help="[EMSANet decoder] Downsampling at the end of each semantic "
                 "decoder module. Length of tuple must match "
                 "`--semantic-decoder-n-channels`."
        )
        group.add_argument(
            '--semantic-decoder-upsampling',
            choices=KNOWN_UPSAMPLING_METHODS,
            default='learned-3x3-zeropad',
            help="How features are upsampled in the semantic decoders. "
                 "Bilinear upsampling may cause problems when converting to "
                 "TensorRT. 'learned-3x3*' mimics bilinear interpolation with "
                 "nearest interpolation and adaptable 3x3 depth-wise "
                 "convolution subsequently (EMSANet decoder only)."
        )

        # -> instance related parameters
        group = self.add_argument_group('Model: Decoder(s) -> Instance')
        group.add_argument(
            '--instance-encoder-decoder-fusion',
            type=str,
            choices=KNOWN_ENCODER_DECODER_FUSIONS,
            default='add-rgb',
            help="Determines how features of the encoder (after fusing "
                 "encoder features) are fused into the instance decoder."
        )
        group.add_argument(
            '--instance-decoder',
            type=str,
            default='emsanet',
            choices=KNOWN_DECODERS,
            help="Decoder type to use for instance segmentation."
        )
        group.add_argument(
            '--instance-decoder-block',
            type=str,
            default='nonbottleneck1d',
            choices=KNOWN_BLOCKS,
            help="[EMSANet decoder] Block (type) to use in instance decoder."
        )
        group.add_argument(
            '--instance-decoder-block-dropout-p',
            type=float,
            default=0.2,
            help="[EMSANet decoder] Dropout probability to use in instance "
                 "decoder blocks (only for 'nonbottleneck1d')."
        )
        group.add_argument(
            '--instance-decoder-n-blocks',
            type=int,
            default=3,
            help="[EMSANet decoder] Number of blocks to use in each instance "
                 "decoder module."
        )
        group.add_argument(
            '--instance-decoder-dropout-p',
            type=float,
            default=0.1,
            help="[SegFormerMLP decoder] Probability to use for feature "
                 "dropout (Dropout2d) in instance decoder before task head."
        )
        group.add_argument(
            '--instance-decoder-n-channels',
            type=int,
            default=(512, 256, 128),
            nargs='+',
            help="[EMSANet decoder] Number of features maps (channels) to use "
                 "in each instance decoder module. Length of tuple "
                 "determines the number of decoder modules. "
                 "[SegFormerMLP decoder] Embedding dimensions to use for main "
                 "branch and skip connections."
        )
        group.add_argument(
            '--instance-decoder-downsamplings',
            type=int,
            default=(16, 8, 4),
            nargs='+',
            help="[EMSANet decoder] Downsampling at the end of each instance "
                 "decoder module. Length of tuple must match "
                 "`--instance-decoder-n-channels`."
        )
        group.add_argument(
            '--instance-decoder-upsampling',
            choices=KNOWN_UPSAMPLING_METHODS,
            default='learned-3x3-zeropad',
            help="How features are upsampled in the instance decoders. "
                 "Bilinear upsampling may cause problems when converting to "
                 "TensorRT. 'learned-3x3*' mimics bilinear interpolation with "
                 "nearest interpolation and adaptable 3x3 depth-wise "
                 "convolution subsequently (EMSANet decoder only)."
        )
        group.add_argument(
            '--instance-center-sigma',
            type=int,
            default=8,
            help="Sigma to use for encoding instance centers. Instance "
                 "centers are encoded in a heatmap using a gauss up to "
                 "3*sigma. Note  that `sigma` is adjusted when using "
                 "multiscale supervision as follows: "
                 "sigma_s = (4*`sigma`) // s for downscale of s."
        )
        group.add_argument(
            '--instance-center-heatmap-threshold',
            type=float,
            default=0.1,
            help="Threshold to use for filtering valid instances during "
                 "postprocessing the predicted center heatmaps. The order of "
                 "postprocessing operations is: threshold, nms, opt. masking, "
                 "top-k."
        )
        group.add_argument(
            '--instance-center-heatmap-nms-kernel-size',
            type=int,
            default=17,
            help="Kernel size for non-maximum suppression to use for "
                 "filtering the predicting instance center heatmaps during "
                 "postprocessing. The order of postprocessing operations is: "
                 "threshold, nms, opt. masking, top-k."
        )
        group.add_argument(
            '--instance-center-heatmap-apply-foreground-mask',
            action='store_true',
            default=False,
            help="Apply foreground mask to centers after non-maximum "
                 "suppression. This filters instance centers that do not "
                 "actually belong to the foreground and, thus, prevents "
                 "instance pixels after offset shifting being assigned to "
                 "such an instance center later on. The order of "
                 "postprocessing operations is: threshold, nms, opt. masking, "
                 "top-k."
        )
        group.add_argument(
            '--instance-center-heatmap-top-k',
            type=int,
            default=64,
            help="Top-k instances to finally select during postprocessing "
                 "instance center heatmaps. The order of postprocessing "
                 "operations is: threshold, nms, opt. masking, top-k.")
        group.add_argument(
            '--instance-center-encoding',
            type=str,
            choices=('deeplab', 'sigmoid'),
            default='sigmoid',
            help="Determines how to encode the predicted instance centers. "
                 "'deeplab' corresponds to simple linear encoding. "
                 "'sigmoid' forces the output to be in range [0., 1.] by "
                 "applying sigmoid activation."
        )
        group.add_argument(
            '--instance-offset-encoding',
            type=str,
            choices=('deeplab', 'relative', 'tanh'),
            default='tanh',
            help="Determines how to encode the predicted instance offset "
                 "vectors. 'deeplab' corresponds to absolute coordinates as "
                 "done in panoptic deeplab."
                 "'relative' means [-1., 1.] with respect to the"
                 "network input resolution. 'tanh is similar to 'relative' "
                 "but further forces [-1., 1.] by applying tanh activation."
                 "Note that this also affects instance target generation.")
        group.add_argument(
            '--instance-offset-distance-threshold',
            type=int,
            default=None,
            help="Distance threshold in pixels to mask out invalid instance "
                 "assignments. Pixels that are more than this threshold away"
                 "from the next instance center after offset shifting, are "
                 "assigned to the 'no instance id' (id=0). Note that this "
                 "masking may lead to thing segments without an instance id, "
                 "which have to be handled later on. During panoptic merging, "
                 "masked pixels are assigned to the void class."
        )

        # -> normal related parameters
        group = self.add_argument_group('Model: Decoder(s) -> Normal')
        group.add_argument(
            '--normal-encoder-decoder-fusion',
            type=str,
            choices=KNOWN_ENCODER_DECODER_FUSIONS,
            default='add-rgb',
            help="Determines how features of the encoder (after fusing "
                 "encoder features) are fused into the normal decoder."
        )
        group.add_argument(
            '--normal-decoder',
            type=str,
            default='emsanet',
            choices=KNOWN_DECODERS,
            help="Decoder type to use for normal segmentation."
        )
        group.add_argument(
            '--normal-decoder-block',
            type=str,
            default='nonbottleneck1d',
            choices=KNOWN_BLOCKS,
            help="[EMSANet decoder] Block (type) to use in normal decoder."
        )
        group.add_argument(
            '--normal-decoder-block-dropout-p',
            type=float,
            default=0.2,
            help="[EMSANet decoder] Dropout probability to use in normal "
                 "decoder blocks (only for 'nonbottleneck1d')."
        )
        group.add_argument(
            '--normal-decoder-n-blocks',
            type=int,
            default=3,
            help="[EMSANet decoder] Number of blocks to use in each normal "
                 "decoder module."
        )
        group.add_argument(
            '--normal-decoder-dropout-p',
            type=float,
            default=0.1,
            help="[SegFormerMLP decoder] Probability to use for feature "
                 "dropout (Dropout2d) in normal decoder before task head."
        )
        group.add_argument(
            '--normal-decoder-n-channels',
            type=int,
            default=(512, 256, 128),
            nargs='+',
            help="[EMSANet decoder] Number of features maps (channels) to use "
                 "in each normal decoder module. Length of tuple "
                 "determines the number of decoder modules. "
                 "[SegFormerMLP decoder] Embedding dimensions to use for main "
                 "branch and skip connections."
        )
        group.add_argument(
            '--normal-decoder-downsamplings',
            type=int,
            default=(16, 8, 4),
            nargs='+',
            help="[EMSANet decoder] Downsampling at the end of each normal "
                 "decoder module. Length of tuple must match "
                 "`--normal-decoder-n-channels`."
        )
        group.add_argument(
            '--normal-decoder-upsampling',
            choices=KNOWN_UPSAMPLING_METHODS,
            default='learned-3x3-zeropad',
            help="How features are upsampled in the normal decoders. "
                 "Bilinear upsampling may cause problems when converting to "
                 "TensorRT. 'learned-3x3*' mimics bilinear interpolation with "
                 "nearest interpolation and adaptable 3x3 depth-wise "
                 "convolution subsequently (EMSANet decoder only)."
        )

        # training ------------------------------------------------------------
        group = self.add_argument_group('Training')
        group.add_argument(
            '--dropout-p',
            type=float,
            default=0.1,
            help="Dropout probability to use in encoder blocks (only for "
                 "'nonbottleneck1d')."
        )
        group.add_argument(
            '--he-init',
            nargs='+',
            type=str,
            choices=('encoder-fusion', 'encoder-decoder-fusion',
                     'context-module',
                     'decoder'),
            default=('encoder-fusion', ),
            help="Initialize weights in given parts of the network using He "
                 "initialization instead of PyTorch's default initialization "
                 "(commonly used heuristic). Note that bias weights are "
                 "allways initialized using pytorch's default (commonly used "
                 "heuristic)."
        )
        group.add_argument(
            '--no-zero-init-decoder-residuals',
            action='store_true',
            default=False,
            help="Disables zero-initializing weights in the last BN in each "
                 "block, so that the residual branch starts with zeros, and "
                 "each residual block behaves like an identity."
        )

        group.add_argument(
            '--n-epochs',
            type=int,
            default=500,
            help="Number of epochs to train for."
        )
        group.add_argument(
            '--batch-size',
            type=int,
            default=8,
            help="Batch size to use for training."
        )
        group.add_argument(
            '--optimizer',
            type=str,
            choices=KNOWN_OPTIMIZERS,
            default='sgd',
            help="Optimizer to use."
        )
        group.add_argument(
            '--learning-rate',
            type=float,
            default=0.01,
            help="Maximum learning rate for a `batch-size` of 8. When using a "
                 "deviating batch size, the learning rate is scaled "
                 "automatically: lr = `learning-rate` * `batch-size`/8."
        )
        group.add_argument(
            '--learning-rate-scheduler',
            type=str,
            choices=KNOWN_LR_SCHEDULERS,
            default='onecycle',
            help="Learning rate scheduler to use. For parameters and details, "
                 "see implementation."
        )
        group.add_argument(
            '--momentum',
            type=float,
            default=0.9,
            help="Momentum to use."
        )
        group.add_argument(
            '--weight-decay',
            type=float,
            default=1e-4,
            help="Weight decay to use for all network weights."
        )
        group.add_argument(
            '--tasks-weighting',
            nargs='+',
            type=float,
            default=None,
            help="Task weighting to use for loss balancing. The tasks' "
                 "weights are assigned to the task in the order given by "
                 "`tasks`."
        )

        # -> semantic related parameters
        group = self.add_argument_group('Training -> Semantic')
        group.add_argument(
            '--semantic-class-weighting',
            type=str,
            choices=KNOWN_CLASS_WEIGHTINGS,
            default='median-frequency',
            help="Weighting mode to use for semantic classes to balance loss "
                 "during training"
        )
        group.add_argument(
            '--semantic-class-weighting-logarithmic-c',
            type=float,
            default=1.02,
            help="Parameter c for limiting the upper bound of the class "
                 "weights when `semantic-class-weighting` is 'logarithmic'. "
                 "Logarithmic class weighting is defined as 1 / ln(c+p_class)."
        )
        group.add_argument(
            "--semantic-loss-label-smoothing",
            type=float,
            default=0.0,
            help="Label smoothing factor to use in loss function for semantic "
                 "segmentation."
        )
        group.add_argument(
            '--semantic-no-multiscale-supervision',
            action='store_true',
            default=False,
            help="Disables multi-scale supervision for semantic decoder."
        )

        # -> instance related parameters
        group = self.add_argument_group('Training -> Instance')
        group.add_argument(
            '--instance-weighting',
            nargs=2,
            type=int,
            default=(2, 1),
            help="Weighting to use for instance task loss balancing with "
                 "format: 'center offset'. The resulting instance task loss "
                 "will then again be weighted with the weight given with "
                 "`tasks-weighting`."
        )
        group.add_argument(
            '--instance-center-loss',
            type=str,
            choices=KNOWN_INSTANCE_CENTER_LOSS_FUNCTIONS,
            default='mse',
            help='Loss function for instance centers.'
        )
        group.add_argument(
            '--instance-no-multiscale-supervision',
            action='store_true',
            default=False,
            help="Disables multi-scale supervision for instance decoder."
        )

        # -> orientation related parameters
        group = self.add_argument_group('Training -> Orientation')
        group.add_argument(
            '--orientation-kappa',
            type=float,
            default=1.0,
            help="Parameter kappa to use for VonMises loss."
        )

        # -> normal related parameters
        group = self.add_argument_group('Training -> Normal')
        group.add_argument(
            '--normal-loss',
            type=str,
            choices=KNOWN_NORMAL_LOSS_FUNCTIONS,
            default='l1',
            help='Loss function for normal.'
        )
        group.add_argument(
            '--normal-no-multiscale-supervision',
            action='store_true',
            default=False,
            help="Disables multi-scale supervision for normal decoder."
        )

        # -> scene related parameters
        group = self.add_argument_group('Training -> Scene')
        group.add_argument(
            "--scene-loss-label-smoothing",
            type=float,
            default=0.1,
            help="Label smoothing factor to use in loss function for scene "
                 "classification."
        )

        # dataset and augmentation --------------------------------------------
        group = self.add_argument_group('Dataset and Augmentation')
        group.add_argument(
            '--dataset',
            type=str,
            default='nyuv2',
            help="Dataset(s) to train/validate on. Use ':' to combine multiple"
                 "datasets. Note that the first dataset is used for "
                 "determining dataset/network/training parameters. Use "
                 "'dataset[camera,camera4]' to select specific cameras. "
                 f"Available datasets: {', '.join(KNOWN_DATASETS)}."
        )
        group.add_argument(
            '--dataset-path',
            type=str,
            default=None,
            help="Path(s) to dataset root(s). If not given, the path is "
                 "determined automatically using the distributed training "
                 "package. If no path(s) can be determined, data loading is "
                 "disabled. Use ':' to combine the paths for combined datasets."
        )
        group.add_argument(
            '--raw-depth',
            action='store_true',
            default=False,
            help="Whether to use the raw depth values instead of refined "
                 "depth values."
        )
        group.add_argument(
            '--use-original-scene-labels',
            action='store_true',
            default=False,
            help="Do not use unified scene class labels for domestic indoor "
                 "environments (Hypersim, NYUv2, ScanNet, and SUNRGB-D only)."
        )
        group.add_argument(
            '--aug-scale-min',
            type=float,
            default=1.0,
            help="Minimum scale for random rescaling during training."
        )
        group.add_argument(
            '--aug-scale-max',
            type=float,
            default=1.4,
            help="Maximum scale for random rescaling during training."
        )
        group.add_argument(
            '--cache-dataset',
            action='store_true',
            default=False,
            help="Cache dataset to speed up training."
        )
        group.add_argument(
            '--n-workers',
            type=int,
            default=8,
            help="Number of workers for data loading and preprocessing"
        )
        group.add_argument(
            '--subset-train',
            type=float,
            default=1.0,
            choices=Range(0.0, 1.0),
            help="Relative value to train on a subset of the train data. For "
                 "example if `subset-train`=0.2 and we have 100 train images, "
                 "then we train only on 20 images. These 20 images are chosen "
                 "randomly each epoch, except if `subset-deterministic` is "
                 "set."
        )
        group.add_argument(
            '--subset-deterministic',
            action='store_true',
            default=False,
            help="Use the same subset in each epoch and across different "
                 "training runs. Requires `subset-train` to be set."
        )
        group = self.add_argument_group('Dataset and Augmentation -> ScanNet')
        # -> ScanNet related parameters
        group.add_argument(
            '--scannet-subsample',
            type=int,
            default=50,
            choices=(50, 100, 200, 500),
            help="Subsample to use for ScanNet dataset for training."
        )
        group.add_argument(
            '--scannet-semantic-n-classes',
            type=int,
            default=40,
            choices=(20, 40, 200, 549),
            help="Number of semantic classes to use for ScanNet dataset."
        )
        # -> SUNRGB-D related parameters
        group = self.add_argument_group('Dataset and Augmentation -> SUNRGB-D')
        group.add_argument(
            '--sunrgbd-depth-do-not-force-mm',
            action='store_true',
            default=False,
            help="Do not force mm for SUNRGB-D depth values. Use this option "
                 "to evaluate weights of the EMSANet paper on SUNRGB-D."
        )
        # -> Hypersim related parameters
        # TODO: can be removed from codebase later
        group = self.add_argument_group('Dataset and Augmentation -> Hypersim')
        group.add_argument(
            '--hypersim-use-old-depth-stats',
            action='store_true',
            default=False,
            help="Use old (v030) depth stats for Hypersim dataset. Enable "
                 "this argument if you load weights created earlier than "
                 "Apr. 28, 2022."
        )
        group.add_argument(
            '--hypersim-subsample',
            type=int,
            default=1,
            choices=(1, 2, 5, 10, 20),
            help="Subsample to use for ScanNet dataset for training."
        )

        # validation/evaluation ------------------------------------------------
        group = self.add_argument_group('Validation/Evaluation')
        group.add_argument(
            '--validation-only',
            action='store_true',
            default=False,
            help="No training, validation only. Requires `weights-filepath`."
        )
        group.add_argument(
            '--visualize-validation',
            default=False,
            action='store_true',
            help="Whether the validation images should be visualized."
        )
        group.add_argument(
            '--visualization-output-path',
            type=str,
            default=None,
            help="Path where to save visualized predictions. By default, a "
                 "new directory is created in the directory where the weights "
                 "come from. The filename of the weights is included in the "
                 "name of the visualization directory, so that it is evident "
                 "which weights have led to these visualizations."
        )
        group.add_argument(    # useful for appm context module
            '--validation-input-height',
            type=int,
            default=None,
            help="Network input height for validation. Images will be resized "
                 "to this height. If not given, `input-height` is used (same "
                 "height for training and validation). "
        )
        group.add_argument(    # useful for appm context module
            '--validation-input-width',
            type=int,
            default=None,
            help="Network input width for validation. Images will be resized "
                 "to this width. If not given, `input-width` is used (same "
                 "width for training and validation)."
        )
        group.add_argument(
            '--validation-batch-size',
            type=int,
            default=None,
            help="Batch size to use for validation. Can be typically 2-3 "
                 "times as large as the batch size for training. If not given "
                 "it will be set to 3 times `batch-size`."
        )
        group.add_argument(
            '--validation-split',
            type=str,
            default='valid',
            help="Dataset split(s) to use for validation. Use ':' to combine "
                 "the splits for combined datasets."
        )
        group.add_argument(
            '--validation-skip',
            type=float,
            default=0.0,
            help="Skip validation (metric calculation, example creation, and "
                 "checkpointing) in early epochs. For example, passing a"
                 "value of '0.2' and `n_epochs` of '500', skips validation "
                 "for the first 0.2*500 = 100 epochs. A value of '1.0' "
                 "disables validation at all."
        )
        group.add_argument(
            '--validation-force-interval',
            type=int,
            default=20,
            help="Force validation after every X epochs even when using "
                 "`validation-skip`. This allows to still see progress and "
                 "save checkpoints during training."
        )
        group.add_argument(
            '--validation-full-resolution',
            action='store_true',
            default=False,
            help="Whether to validate on full-resolution inputs (do not apply "
                 "any resizing to the inputs, for Cityscapes or "
                 "Hypersim dataset)."
        )
        # -> ScanNet related parameters
        group = self.add_argument_group('Validation/Evaluation -> ScanNet')
        group.add_argument(
            '--validation-scannet-subsample',
            type=int,
            default=100,
            choices=(50, 100, 200, 500),
            help="Subsample to use for ScanNet dataset for validation."
        )
        group.add_argument(
            '--validation-scannet-benchmark-mode',
            action='store_true',
            default=False,
            help="Enable benchmark mode for validation on ScanNet dataset, "
                 "i.e., mapping ignored classes to void "
                 "(`scannet-semantic-n-classes`=40/549 only)."
        )
        # -> checkpointing
        group = self.add_argument_group(
            'Validation/Evaluation -> Checkpointing'
        )
        group.add_argument(
            '--checkpointing-metrics',
            nargs='+',
            type=str,
            default=None,
            help="Metric(s) to use for checkpointing. For example "
                 "'miou bacc miou+bacc' leads to checkpointing when either "
                 "miou, bacc, or the sum of miou and bacc reaches its highest "
                 "value. Note that current implemention only supports "
                 "combining metrics using '+'. Omitted this parameter "
                 "disables checkpointing."
        )
        group.add_argument(
            '--checkpointing-best-only',
            action='store_true',
            default=False,
            help="Store only the best checkpoint."
        )
        group.add_argument(
            '--checkpointing-skip',
            type=float,
            default=0.0,
            help="Skip checkpointing in early epochs. For example, passing a"
                 "value of '0.2' and `n_epochs` of '500', skips checkpointing "
                 "for the first 0.2*500 = 100 epochs. A value of '1.0' "
                 "disables checkpointing at all."
        )

        # resuming ------------------------------------------------------------
        subparsers = self.add_subparsers(
            parser_class=ap.ArgumentParser,     # important to avoid recursion
            # required=False,    # python >= 3.7 feature
            dest='action'
        )
        subparser = subparsers.add_parser(
            'resume',
            help="Resume previous training run with auto argument and "
                 "checkpoint detection, see `path` argument for details."
        )
        subparser.add_argument(
            'resume_path',
            type=str,
            default=None,
            help="Path to previous training run, e.g., './runs_xy'. All args "
                 "are automatically replaced with the args given in "
                 "'./runs_xy/argsv.txt'. Furthermore, `--resume-ckpt-filepath` "
                 "is set to './runs_xy/checkpoints/ckpt_resume.pth'. For "
                 "safety, a backup of the given run folder is created."
        )

        self.add_argument(
            '--resume-ckpt-filepath',
            type=str,
            default=None,
            help="Path to checkpoint file to resume training from. "
        )

        self.add_argument(
            '--resume-ckpt-interval',
            type=int,
            default=20,
            help="Write resume checkpoint containing state dicts for model, "
                 "optimizer, and lr scheduler every X epochs. "
                 "This allows resuming a previous training."
        )

        # debugging -----------------------------------------------------------
        self.add_argument(
            '--debug',
            action='store_true',
            default=False,
            help="Enables debug outputs (and exporting the model to ONNX)."
        )
        self.add_argument(
            '--skip-sanity-check',
            action='store_true',
            default=False,
            help="Disables the simple sanity check before training that "
                 "ensures that crucial parts (data, forward, metrics, ...) are "
                 "working as expected. The check is done by forwarding a "
                 "single batch of all dataloadera WITHOUT backpropagation."
        )
        self.add_argument(
            '--overfit-n-batches',
            type=int,
            default=-1,
            help="Forces to overfit on specified number of batches. Note that "
                 "for both training and validation samples are drawn from the "
                 "(first) validation loader without shuffling."
        )
        # Weights & Biases ----------------------------------------------------
        self.add_argument(
            '--wandb-mode',
            type=str,
            choices=('online', 'offline', 'disabled'),     # see wandb
            default='online',
            help="Mode for Weights & Biases"
        )
        self.add_argument(
            '--wandb-project',
            type=str,
            default='EMSANet',
            help="Project name for Weights & Biases"
        )
        self.add_argument(
            '--wandb-name',
            type=str,
            default=None,
            help="[DEPRECATED] Use `--wandb-project` instead."
        )

        # other parameters ----------------------------------------------------
        self.add_argument(
            '--hostname',
            type=str,
            default=socket.gethostname(),
            help="We are often interested in the hostname the code is running."
        )
        self.add_argument(
            '--notes',
            type=str,
            default='',
            help="Just to add some additional notes for this run."
        )

    def parse_args(self, args=None, namespace=None, verbose=True):
        # parse args
        pa = super().parse_args(args=args, namespace=namespace)

        def _warn(text):
            if verbose:
                print(f"[Warning] {text}")

        # check for resumed training ------------------------------------------
        if 'resume' == pa.action:
            is_resumed_training = True
            resume_path = pa.resume_path

            # load args from file
            args_fp = os.path.join(pa.resume_path, 'argsv.txt')
            print(f"Resuming training with args from: '{args_fp}'.")
            with open(args_fp, 'r') as f:
                args_str = f.read().strip()
            args_run = shlex.split(args_str)[1:]     # remove script name

            # create a backup of the given run folder
            backup_number = 1
            while 0 != backup_number:    # was not successful
                backup_path = os.path.normpath(pa.resume_path)  # trailing /
                backup_path += f'_before_resume{backup_number}'

                if os.path.isdir(backup_path):
                    print(f"Found already existing backup: '{backup_path}'")
                    backup_number += 1
                    continue

                print(f"Creating backup at: '{backup_path}'.")
                shutil.copytree(src=pa.resume_path, dst=backup_path)
                break

            # set resume checkpoint filepath
            ckpt_fp = os.path.join(pa.resume_path, 'checkpoints',
                                   'ckpt_resume.pth')
            assert os.path.isfile(ckpt_fp)
            args_run.extend(
                shlex.split(f'--resume-ckpt-filepath {shlex.quote(ckpt_fp)}')
            )
            # parse args again
            pa = super().parse_args(args=args_run, namespace=namespace)
        else:
            is_resumed_training = False
            resume_path = None

        # store additional information
        pa.resume_path = resume_path
        pa.is_resumed_training = is_resumed_training

        # convert nargs+ arguments from lists to tuples -----------------------
        for k, v in dict(vars(pa)).items():
            if isinstance(v, list):
                setattr(pa, k, tuple(v))

        # perform some initial argument checks
        # weights filepaths ---------------------------------------------------
        if pa.encoder_backbone_pretrained_weights_filepath is not None:
            # check if filepaths for rgb and depth are not set
            if any((pa.rgb_encoder_backbone_pretrained_weights_filepath is not None,
                    pa.depth_encoder_backbone_pretrained_weights_filepath is not None,
                    pa.rgbd_encoder_backbone_pretrained_weights_filepath is not None)):
                raise ValueError(
                    "Only use `encoder-backbone-pretrained-weights-filepath` "
                    "if you want to initialize all used encoder backbones with "
                    "the same weights! "
                    "`rgb-encoder-backbone-pretrained-weights-filepath` and "
                    "`depth-encoder-backbone-pretrained-weights-filepath` and "
                    "`rgbd-encoder-backbone-pretrained-weights-filepath` must"
                    "not be set."
                )
            pa.rgb_encoder_backbone_pretrained_weights_filepath = \
                pa.encoder_backbone_pretrained_weights_filepath
            pa.depth_encoder_backbone_pretrained_weights_filepath = \
                pa.encoder_backbone_pretrained_weights_filepath
            pa.rgbd_encoder_backbone_pretrained_weights_filepath = \
                pa.encoder_backbone_pretrained_weights_filepath
        # this argument is not needed anymore
        del pa.encoder_backbone_pretrained_weights_filepath

        # model ---------------------------------------------------------------
        # handle deprecated normalization choice for whole model
        if pa.normalization is not None:
            pa.encoder_normalization = pa.normalization
            pa.decoder_normalization = pa.normalization
            _warn("Forced `encoder-normalization` and `decoder-normalization`, "
                  f"to be '{pa.normalization}' as `normalization` was given.")
        # handle deprecated resnet block choice for encoder
        if pa.rgb_encoder_backbone_block is not None:
            pa.rgb_encoder_backbone_resnet_block = pa.rgb_encoder_backbone_block
            _warn("Forced `rgb-encoder-backbone-resnet-block`, to be "
                  f"'{pa.rgb_encoder_backbone_block}' as "
                  "`rgb-encoder-backbone-block` was given.")
        if pa.depth_encoder_backbone_block is not None:
            pa.depth_encoder_backbone_resnet_block = pa.depth_encoder_backbone_block
            _warn("Forced `depth-encoder-backbone-resnet-block`, to be "
                  f"'{pa.depth_encoder_backbone_block}' as "
                  "`depth-encoder-backbone-block` was given.")

        # handle deprecated encoder-decoder-fusion
        if pa.encoder_decoder_fusion is not None:
            pa.semantic_encoder_decoder_fusion = pa.encoder_decoder_fusion
            _warn("Forced `semantic-encoder-decoder-fusion`, to be "
                  f"'{pa.encoder_decoder_fusion}' as `encoder-decoder-fusion` "
                  "was given.")
            pa.instance_encoder_decoder_fusion = pa.encoder_decoder_fusion
            _warn("Forced `semantic-encoder-decoder-fusion`, to be "
                  f"'{pa.encoder_decoder_fusion}' as `encoder-decoder-fusion` "
                  "was given.")
            pa.normal_encoder_decoder_fusion = pa.encoder_decoder_fusion
            _warn("Forced `normal-encoder-decoder-fusion`, to be "
                  f"'{pa.encoder_decoder_fusion}' as `encoder-decoder-fusion` "
                  "was given.")

        # handle deprecated upsampling-decoder
        if pa.upsampling_decoder is not None:
            pa.semantic_decoder_upsampling = pa.upsampling_decoder
            _warn("Forced `semantic-decoder-upsampling`, to be "
                  f"'{pa.upsampling_decoder}' as `upsampling-decoder` "
                  "was given.")
            pa.instance_decoder_upsampling = pa.upsampling_decoder
            _warn("Forced `instance-decoder-upsampling`, to be "
                  f"'{pa.upsampling_decoder}' as `upsampling-decoder` "
                  "was given.")
            pa.normal_decoder_upsampling = pa.upsampling_decoder
            _warn("Forced `normal-decoder-upsampling`, to be "
                  f"'{pa.upsampling_decoder}' as `upsampling-decoder` "
                  "was given.")

        # disable encoder fusion if only one input modality is used
        if 1 == len(pa.input_modalities):
            pa.encoder_fusion = 'none'
            _warn("Set `encoder-fusion` to 'none' as there is only one input "
                  "modality.")

        # multi-task parameters -----------------------------------------------
        if 'orientation' in pa.tasks:
            if 'instance' not in pa.tasks:
                raise ValueError("Task 'instance' is missing in `tasks` for "
                                 "performing task 'orientation'.")

        if pa.enable_panoptic:
            if 'semantic' not in pa.tasks:
                raise ValueError("Task 'semantic' is missing in `tasks` for "
                                 "performing panoptic segmentation.")

            if 'instance' not in pa.tasks:
                raise ValueError("Task 'instance' is missing in `tasks` for "
                                 "performing panoptic segmentation.")
        # training ------------------------------------------------------------
        if pa.batch_size != 8:
            # the provided learning rate refers to the default batch size of 8
            # when using different batch sizes, we need to adjust the learning
            # rate accordingly
            pa.learning_rate = pa.learning_rate * pa.batch_size / 8
            _warn(f"Adapted learning rate to '{pa.learning_rate}' as the "
                  f"provided batch size differs from default batch size of 8.")

        if pa.tasks_weighting is None:
            # default weighting (required for inference or timing)
            pa.tasks_weighting = (1,)*len(pa.tasks)

        if len(pa.tasks_weighting) != len(pa.tasks):
            raise ValueError("Length for given task weighting does not match "
                             f"number of tasks: {len(pa.tasks_weighting)} vs. "
                             f"{len(pa.tasks)}.")

        # common failures for ScanNet
        if 'scannet' in pa.dataset and pa.validation_scannet_benchmark_mode:
            if pa.scannet_semantic_n_classes not in (40, 549):
                raise ValueError(
                    "`validation-scannet-benchmark-mode` requires "
                    "`scannet-semantic-n-classes` to be 40 or 549."
                )

        # common failures for COCO
        if 'coco' in pa.dataset:
            if 'depth' in pa.input_modalities:
                raise ValueError("COCO dataset does not feature depth data.")
            if 'normal' in pa.tasks:
                raise ValueError("COCO dataset does not feature surface "
                                 "normals.")
            if 'scene' in pa.tasks:
                raise ValueError("Scene classification is not supported for "
                                 "COCO dataset.")

        if any(d in pa.dataset for d in ('cityscapes', 'hypersim', 'scannet')):
            # Depth data for hypersim is clipped to the limit of png16 (uint16)
            # during dataset preparation. To account for that and to ignore
            # these pixels '--raw-depth' should be forced. Note, the actual
            # amount of clipped pixels is quite small.
            pa.raw_depth = True
            _warn(f"Forced `raw-depth` as `dataset` is '{pa.dataset}'.")

        # check whether provided decoder type supports multiscale supervision
        decoders_with_ms = ('emsanet',)
        if pa.semantic_decoder not in decoders_with_ms:
            if not pa.semantic_no_multiscale_supervision:
                pa.semantic_no_multiscale_supervision = True
                _warn("Forced `semantic-no-multiscale-supervision` as "
                      f"`semantic-decoder` is '{pa.semantic_decoder}'.")
        if pa.instance_decoder not in decoders_with_ms:
            if not pa.instance_no_multiscale_supervision:
                pa.instance_no_multiscale_supervision = True
                _warn("Forced `instance-no-multiscale-supervision` as "
                      f"`instance-decoder` is '{pa.instance_decoder}'.")
        if pa.normal_decoder not in decoders_with_ms:
            if not pa.normal_no_multiscale_supervision:
                pa.normal_no_multiscale_supervision = True
                _warn("Forced `normal-no-multiscale-supervision` as "
                      f"`normal-decoder` is '{pa.normal_decoder}'.")

        # evaluation ----------------------------------------------------------
        if pa.validation_full_resolution:
            if not any(d in pa.dataset for d in ('cityscapes', 'hypersim')):
                # height/width in cityscapes and hypersim are multiple 32
                raise ValueError(
                    "Validation with full resolution inputs is only supported"
                    "for 'cityscapes' or 'hypersim'."
                )
        # ensure that validation input size is set (None -> input size)
        if pa.validation_input_width is None:
            pa.validation_input_width = pa.input_width
        if pa.validation_input_height is None:
            pa.validation_input_height = pa.input_height

        if pa.validation_batch_size is None:
            pa.validation_batch_size = 3*pa.batch_size
            _warn(f"`validation-batch-size` not given, using default: "
                  f"{pa.validation_batch_size}.")

        # handle some common misconfigurations
        if 'valid' == pa.validation_split and pa.dataset in ('nyuv2',
                                                             'sunrgbd'):
            pa.validation_split = 'test'
            _warn(f"Dataset '{pa.dataset}' does not have a 'valid' split, "
                  "using 'test' split instead.")

        if pa.validation_skip > pa.checkpointing_skip:
            _warn(f"Setting `checkpointing_skip` to '{pa.validation_skip}' as "
                  f"`validation_skip` is larger '{pa.validation_skip}'.")

        # set default for pa.visualization_output_path and check that it does
        # not already exist
        if pa.visualize_validation:
            if pa.visualization_output_path is None:
                weights_dirpath, weights_filename = os.path.split(
                    pa.weights_filepath
                )
                pa.visualization_output_path = os.path.join(
                    weights_dirpath,
                    f'visualization_{os.path.splitext(weights_filename)[0]}'
                )
            if os.path.exists(pa.visualization_output_path):
                raise ValueError(
                    "The path provided by `visualization-output-path` "
                    f"'{pa.visualization_output_path}' already exists. Please "
                    "provide a different path."
                )

        # TODO: can be removed from codebase later
        if all((pa.validation_only,
                'hypersim' in pa.dataset,
                pa.weights_filepath is not None,
                not pa.hypersim_use_old_depth_stats)):

            from datetime import datetime

            t_weights_created = os.path.getctime(pa.weights_filepath)
            t_fix = datetime.strptime('2022.04.28', '%Y.%m.%d').timestamp()
            if t_weights_created < t_fix:
                _warn("Detected Hypersim checkpoint created earlier than "
                      "Apr. 28, 2022. Consider adding "
                      "`hypersim-use-old-depth-stats` argument to ensure "
                      "correct depth stats.")

        # other parameters ----------------------------------------------------
        if pa.debug:
            _warn("`debug` is set, enabling debug outputs and ONNX export "
                  "(use EXPORT_ONNX_MODELS=true python ...)")

        # wandb
        if pa.wandb_name is not None:
            _warn("Parameter `wandb-name` is deprecated, use `wandb-project` "
                  "instead.")
            pa.wandb_project = pa.wandb_name

        # print args
        if verbose:
            args_str = json.dumps(vars(pa), indent=4, sort_keys=True)
            print(f"Running with args:\n {args_str}")

        # return parsed (and subsequently modified) args
        return pa
