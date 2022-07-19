# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
"""
import json
import os
import argparse as ap
import warnings

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
from .lr_scheduler import KNOWN_LR_SCHEDULERS
from .optimizer import KNOWN_OPTIMIZERS


class Range(object):
    """
    helper for argparse to restrict floats to be in a specified range.
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

        # paths ----------------------------------------------------------------
        self.add_argument(
             '--results-basepath',
             type=str,
             default='./results',
             help="Path where to store training files."
        )
        self.add_argument(
            '--encoder-backbone-pretrained-weights-filepath',
            type=str,
            default=None,
            help="Path to pretrained (ImageNet) weights for the encoder "
                 "backbones. Use this argument if you want to initialize both "
                 "encoder backbones with the same weights. "
                 "If `weights-filepath` is given, the specified weights are "
                 "loaded subsequently and may replace the pretrained weights."
        )
        self.add_argument(
            '--rgb-encoder-backbone-pretrained-weights-filepath',
            type=str,
            default=None,
            help="Path to pretrained (ImageNet) weights for the rgb encoder "
                 "backbone. "
                 "If `weights-filepath` is given, the specified weights are "
                 "loaded subsequently and may replace the pretrained weights."
        )
        self.add_argument(
            '--depth-encoder-backbone-pretrained-weights-filepath',
            type=str,
            default=None,
            help="Path to pretrained (ImageNet) weights for the depth encoder "
                 "backbone. "
                 "If `weights-filepath` is given, the specified weights are "
                 "loaded subsequently and may replace the pretrained weights."
        )
        self.add_argument(
            '--weights-filepath',
            type=str,
            default=None,
            help="Filepath to (last) checkpoint / weights to load on start. "
                 "Note that no state dict is loaded and training will still "
                 "start from epoch 0 after loading weights."
        )
        self.add_argument(
            '--visualization-output-path',
            type=str,
            default=None,
            help="Path where to save visualized predictions. By default, a new "
                 "directory is created in the directory where the weights come "
                 "from. The filename of the weights is included in the name of "
                 "the visualization directory, so that it is evident which "
                 "weights have led to these visualizations."
        )

        # network and multi-task -----------------------------------------------
        self.add_argument(
            '--input-height',
            type=int,
            default=480,
            help="Network input height. Images will be resized to this height."
        )
        self.add_argument(
            '--input-width',
            type=int,
            default=640,
            help="Network input width. Images will be resized to this width."
        )
        self.add_argument(
            '--input-modalities',
            nargs='+',
            type=str,
            choices=('rgb', 'depth'),
            default=('rgb', 'depth'),
            help="Input modalities to consider."
        )

        # -> whole model
        self.add_argument(
            '--normalization',
            type=str,
            default='batchnorm',
            choices=KNOWN_NORMALIZATIONS,
            help="Normalization to apply in the whole model."
        )
        self.add_argument(
            '--activation',
            type=str,
            default='relu',
            choices=KNOWN_ACTIVATIONS,
            help="Activation to use in the whole model."
        )
        self.add_argument(
            '--dropout-p',
            type=float,
            default=0.1,
            help="Dropout probability to use in encoder blocks (only for "
                 "'nonbottleneck1d')."
        )

        # -> encoder related parameters
        self.add_argument(
            '--rgb-encoder-backbone',
            type=str,
            choices=KNOWN_BACKBONES,
            default='resnet34',
            help="Backbone to use for RGB encoder."
        )
        self.add_argument(
            '--rgb-encoder-backbone-block',
            type=str,
            choices=KNOWN_BLOCKS,
            default='nonbottleneck1d',
            help="Block (type) to use in RGB encoder backbone."
        )
        self.add_argument(
            '--depth-encoder-backbone',
            type=str,
            choices=KNOWN_BACKBONES,
            default='resnet34',
            help="Backbone to use for depth encoder."
        )
        self.add_argument(
            '--depth-encoder-backbone-block',
            type=str,
            choices=KNOWN_BLOCKS,
            default='nonbottleneck1d',
            help="Block (type) to use in depth encoder backbone."
        )
        self.add_argument(
            '--encoder-fusion',
            choices=KNOWN_ENCODER_FUSIONS,
            default='se-add-uni-rgb',
            help="Determines how features of the depth (rgb) encoder are fused "
                 "to features of the other encoder. Uni- or birectional"
        )

        # -> context module related parameters
        self.add_argument(
            '--context-module',
            type=str,
            choices=KNOWN_CONTEXT_MODULES,
            default='ppm',
            help='Context module to use.'
        )
        self.add_argument(
            '--upsampling-context-module',
            choices=('nearest', 'bilinear'),
            default='bilinear',
            help="How features are upsampled in the context module. Bilinear "
                 "upsampling may cause problems when converting to TensorRT."
        )

        # -> decoder related parameters
        self.add_argument(
            '--encoder-decoder-skip-downsamplings',
            nargs='+',
            type=int,
            default=(4, 8, 16),
            help="Determines at which downsamplings skip connections from the "
                 "encoder to the decoder should be created, e.g., '4, 8' means "
                 "skip connections from the last volume at 1/4 and 1/8 of the "
                 "input size to the decoder."
        )
        self.add_argument(
            '--encoder-decoder-fusion',
            type=str,
            choices=KNOWN_ENCODER_DECODER_FUSIONS,
            default='add-rgb',
            help="Determines how features of the encoder (after fusing encoder "
                 "features) are fused into the decoder."
        )
        self.add_argument(
            '--upsampling-decoder',
            choices=KNOWN_UPSAMPLING_METHODS,
            default='learned-3x3-zeropad',
            help="How features are upsampled in the decoder. Bilinear "
                 "upsampling may cause problems when converting to TensorRT. "
                 "'learned-3x3*' mimics bilinear interpolation with nearest "
                 "interpolation and adaptable 3x3 depth-wise conv subsequently."
        )
        self.add_argument(
            '--upsampling-prediction',
            choices=KNOWN_UPSAMPLING_METHODS,
            default='learned-3x3-zeropad',
            help="How features are upsampled after the last decoder module to "
                 "match the NETWORK input resolution. Bilinear upsampling may "
                 "cause problems when converting to TensorRT. 'learned-3x3*' "
                 "mimics bilinear interpolation with nearest interpolation and "
                 "adaptable 3x3 depth-wise conv subsequently."
        )

        # -> multi-task parameters
        self.add_argument(
            '--tasks',
            nargs='+',
            type=str,
            choices=KNOWN_TASKS,
            default=('semantic',),
            help="Task(s) to perform."
        )

        self.add_argument(
            '--enable-panoptic',
            action='store_true',
            default=False,
            help="Enforces tasks 'semantic' and 'instance' to be combined for "
                 "panoptic segmentation"
        )

        # -> semantic related parameters
        self.add_argument(
            '--semantic-decoder-block',
            type=str,
            default='nonbottleneck1d',
            choices=KNOWN_BLOCKS,
            help="Block (type) to use in semantic decoder."
        )
        self.add_argument(
            '--semantic-decoder-block-dropout-p',
            type=float,
            default=0.2,
            help="Dropout probability to use in semantic decoder blocks (only "
                 "for 'nonbottleneck1d')."
        )
        self.add_argument(
            '--semantic-decoder-n-blocks',
            type=int,
            default=3,
            help="Number of blocks to use in each semantic decoder module."
        )
        self.add_argument(
            '--semantic-decoder-n-channels',
            type=int,
            default=(512, 256, 128),
            help="Number of features maps (channels) to use in each semantic "
                 "decoder module. Length of tuple determines the number of "
                 "decoder modules."
        )

        # -> instance related parameters
        self.add_argument(
            '--instance-decoder-block',
            type=str,
            default='nonbottleneck1d',
            choices=KNOWN_BLOCKS,
            help="Block (type) to use in instance decoder."
        )
        self.add_argument(
            '--instance-decoder-block-dropout-p',
            type=float,
            default=0.2,
            help="Dropout probability to use in instance decoder blocks (only "
                 "for 'nonbottleneck1d')."
        )
        self.add_argument(
            '--instance-decoder-n-blocks',
            type=int,
            default=3,
            help="Number of blocks to use in each instance decoder module."
        )
        self.add_argument(
            '--instance-decoder-n-channels',
            type=int,
            default=(512, 256, 128),
            help="Number of features maps (channels) to use in each instance "
                 "decoder module. Length of tuple determines the number of "
                 "decoder modules."
        )
        self.add_argument(
            '--instance-center-sigma',
            type=int,
            default=8,
            help="Sigma to use for encoding instance centers. Instance centers "
                 "are encoded in a heatmap using a gauss up to 3*sigma. Note "
                 "that `sigma` is adjusted when using multiscale supervision "
                 "as follows: sigma_s = (4*`sigma`) // s for downscale of s.")

        self.add_argument(
            '--instance-center-heatmap-threshold',
            type=float,
            default=0.1,
            help="Threshold to use for filtering valid instances during "
                 "postprocessing the predicted center heatmaps. The order of "
                 "postprocessing operations is: threshold, nms, top-k.")

        self.add_argument(
            '--instance-center-heatmap-nms-kernel-size',
            type=int,
            default=17,
            help="Kernel size for non-maximum suppression to use for filtering "
                 "the predicting instance center heatmaps during "
                 "postprocessing. The order of postprocessing operations is: "
                 "threshold, nms, top-k.")

        self.add_argument(
            '--instance-center-heatmap-top-k',
            type=int,
            default=64,
            help="Top-k instances to finally select during postprocessing "
                 "instance center heatmaps. The order of postprocessing "
                 "operations is: threshold, nms, top-k.")

        self.add_argument(
            '--instance-center-encoding',
            type=str,
            choices=('deeplab', 'sigmoid'),
            default='sigmoid',
            help="Determines how to encode the predicted instance centers. "
                 "'deeplab' corresponds to simple linear encoding. "
                 "'sigmoid' forces the output to be in range [0., 1.] by "
                 "applying sigmoid activation."
        )

        self.add_argument(
            '--instance-offset-encoding',
            type=str,
            choices=('deeplab', 'relative', 'tanh'),
            default='tanh',
            help="Determines how to encode the predicted instance offset "
                 "vectors. 'deeplab' corresponds to absolute coordinates as "
                 "done in panoptic deeplab."
                 "'relative' means [-1., 1.] with respect to the"
                 "network input resolution. 'tanh is similar to 'relative' but "
                 "further forces [-1., 1.] by applying tanh activation."
                 "Note that this also affects instance target generation.")

        # -> normal related parameters
        self.add_argument(
            '--normal-decoder-block',
            type=str,
            default='nonbottleneck1d',
            choices=KNOWN_BLOCKS,
            help="Block (type) to use in normal decoder."
        )
        self.add_argument(
            '--normal-decoder-block-dropout-p',
            type=float,
            default=0.2,
            help="Dropout probability to use in normal decoder blocks (only "
                 "for 'nonbottleneck1d')."
        )
        self.add_argument(
            '--normal-decoder-n-blocks',
            type=int,
            default=3,
            help="Number of blocks to use in each normal decoder module."
        )
        self.add_argument(
            '--normal-decoder-n-channels',
            type=int,
            default=(512, 256, 128),
            help="Number of features maps (channels) to use in each normal "
                 "decoder module. Length of tuple determines the number of "
                 "decoder modules."
        )

        # training -------------------------------------------------------------
        self.add_argument(
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
        self.add_argument(
            '--no-zero-init-decoder-residuals',
            action='store_true',
            default=False,
            help="Disables zero-initializing weights in the last BN in each "
                 "block, so that the residual branch starts with zeros, and "
                 "each residual block behaves like an identity."
        )
        self.add_argument(
            '--no-pretrained-backbone',
            action='store_true',
            default=False,
            help="Disables loading of ImageNet pretrained weights for the "
                 "backbone(s). Useful for inference or inference timing."
        )
        self.add_argument(
            '--n-epochs',
            type=int,
            default=500,
            help="Number of epochs to train for."
        )
        self.add_argument(
            '--batch-size',
            type=int,
            default=8,
            help="Batch size to use for training."
        )
        self.add_argument(
            '--optimizer',
            type=str,
            choices=KNOWN_OPTIMIZERS,
            default='sgd',
            help="Optimizer to use."
        )
        self.add_argument(
            '--learning-rate',
            type=float,
            default=0.01,
            help="Maximum learning rate for a `batch-size` of 8. When using a "
                 "deviating batch size, the learning rate is scaled "
                 "automatically: lr = `learning-rate` * `batch-size`/8."
        )
        self.add_argument(
            '--learning-rate-scheduler',
            type=str,
            choices=KNOWN_LR_SCHEDULERS,
            default='onecycle',
            help="Learning rate scheduler to use. For parameters and details, "
                 "see implementation."
        )
        self.add_argument(
            '--momentum',
            type=float,
            default=0.9,
            help="Momentum to use."
        )
        self.add_argument(
            '--weight-decay',
            type=float,
            default=1e-4,
            help="Weight decay to use for all network weights."
        )
        self.add_argument(
            '--tasks-weighting',
            nargs='+',
            type=float,
            default=None,
            help="Task weighting to use for loss balancing. The tasks' weights "
                 "are assigned to the task in the order given by `tasks`"
        )

        # -> semantic related parameters
        self.add_argument(
            '--semantic-class-weighting',
            type=str,
            choices=KNOWN_CLASS_WEIGHTINGS,
            default='median-frequency',
            help="Weighting mode to use for semantic classes to balance loss "
                 "during training"
        )
        self.add_argument(
            '--semantic-class-weighting-logarithmic-c',
            type=float,
            default=1.02,
            help="Parameter c for limiting the upper bound of the class "
                 "weights when `semantic-class-weighting` is 'logarithmic'. "
                 "Logarithmic class weighting is defined as 1 / ln(c+p_class)."
        )
        self.add_argument(
            "--semantic-loss-label-smoothing",
            type=float,
            default=0.0,
            help="Label smoothing factor to use in loss function for semantic "
                 "segmentation."
        )
        self.add_argument(
            '--semantic-no-multiscale-supervision',
            action='store_true',
            default=False,
            help="Disables multi-scale supervision for semantic decoder."
        )

        # -> instance related parameters
        self.add_argument(
            '--instance-weighting',
            nargs=2,
            type=int,
            default=(2, 1),
            help="Weighting to use for instance task loss balancing with "
                 "format: 'center offset'. The resulting instance task loss "
                 "will then again be weighted with the weight given with "
                 "`tasks-weighting`."
        )
        self.add_argument(
            '--instance-center-loss',
            type=str,
            choices=KNOWN_INSTANCE_CENTER_LOSS_FUNCTIONS,
            default='mse',
            help='Loss function for instance centers.'
        )
        self.add_argument(
            '--instance-no-multiscale-supervision',
            action='store_true',
            default=False,
            help="Disables multi-scale supervision for instance decoder."
        )

        # -> orientation related parameters
        self.add_argument(
            '--orientation-kappa',
            type=float,
            default=1.0,
            help="Parameter kappa to use for VonMises loss."
        )

        # -> normal related parameters
        self.add_argument(
            '--normal-loss',
            type=str,
            choices=KNOWN_NORMAL_LOSS_FUNCTIONS,
            default='l1',
            help='Loss function for normal.'
        )
        self.add_argument(
            '--normal-no-multiscale-supervision',
            action='store_true',
            default=False,
            help="Disables multi-scale supervision for normal decoder."
        )

        # -> scene related parameters
        self.add_argument(
            "--scene-loss-label-smoothing",
            type=float,
            default=0.1,
            help="Label smoothing factor to use in loss function for scene "
                 "classification."
        )

        # dataset and augmentation ---------------------------------------------
        self.add_argument(
            '--dataset',
            type=str,
            choices=KNOWN_DATASETS,
            default='nyuv2',
            help="Dataset to train on."
        )
        self.add_argument(
            '--dataset-path',
            type=str,
            default=None,
            help="Path to dataset root. If not given, the path is determined "
                 "automatically using the distributed training package. If no "
                 "path can be determined, data loading is disabled."
        )
        self.add_argument(
            '--raw-depth',
            action='store_true',
            default=False,
            help="Whether to use the raw depth values instead of refined "
                 "depth values."
        )
        self.add_argument(
            '--aug-scale-min',
            type=float,
            default=1.0,
            help="Minimum scale for random rescaling during training."
        )
        self.add_argument(
            '--aug-scale-max',
            type=float,
            default=1.4,
            help="Maximum scale for random rescaling during training."
        )
        self.add_argument(
            '--cache-dataset',
            action='store_true',
            default=False,
            help="Cache dataset to speed up training."
        )
        self.add_argument(
            '--n-workers',
            type=int,
            default=8,
            help="Number of workers for data loading and preprocessing"
        )
        self.add_argument(
            '--subset-train',
            type=float,
            default=1.0,
            choices=Range(0.0, 1.0),
            help="Relative value to train on a subset of the train data. For "
                 "example if `subset-train`=0.2 and we have 100 train images, "
                 "then we train only on 20 images. These 20 images are chosen "
                 "randomly each epoch, except if `subset-deterministic` is set."
        )
        self.add_argument(
            '--subset-deterministic',
            action='store_true',
            default=False,
            help="Use the same subset in each epoch and across different "
                 "training runs. Requires `subset-train` to be set."
        )
        # TODO: can be removed from codebase later
        self.add_argument(
            '--hypersim-use-old-depth-stats',
            action='store_true',
            default=False,
            help="Use old (v030) depth stats for Hypersim dataset. Enable "
                 "this argument if you load weights created earlier than "
                 "Apr. 28, 2022."
        )

        # evaluation -----------------------------------------------------------
        self.add_argument(
            '--validation-batch-size',
            type=int,
            default=None,
            help="Batch size to use for validation. Can be typically 2-3 times "
                 "as large as the batch size for training. If not given it "
                 "will be set to 3 times `batch-size`."
        )
        self.add_argument(
            '--validation-split',
            type=str,
            default='valid',
            help="Dataset split to use for validation."
        )
        self.add_argument(
            '--validation-only',
            action='store_true',
            default=False,
            help="No training, validation only. Requires `weights-filepath`."
        )
        self.add_argument(
            '--validation-full-resolution',
            action='store_true',
            default=False,
            help="Whether to validate on full-resolution inputs (do not apply "
                 "any resizing to the inputs) as well (for cityscapes or "
                 "hypersim dataset)."
        )
        self.add_argument(
            '--validation-skip',
            type=float,
            default=0.0,
            help="Skip validation (metric calculation, example creation, and "
                 "checkpointing) in early epochs. For example, passing a"
                 "value of '0.2' and `n_epochs` of '500', skips validation "
                 "for the first 0.2*500 = 100 epochs. A value of '1.0' "
                 "disables validation at all."
        )
        self.add_argument(
            '--validation-force-interval',
            type=int,
            default=20,
            help="Force validation after every X epochs even when using "
                 "`validation-skip`. This allows to still see progress and "
                 "save checkpoints during training."
        )
        self.add_argument(
            '--visualize-validation',
            default=False,
            action='store_true',
            help="Wether the validation images should be visualized."
        )
        self.add_argument(
            '--checkpointing-metrics',
            nargs='+',
            type=str,
            default=None,
            help="Metric(s) to use for checkpointing. For example "
                 "'miou bacc miou+bacc' leads to checkpointing when either "
                 "miou, bacc, or the sum of miou and bacc reaches its highest "
                 "value. Note that current implemention only supports "
                 "combining metrics using '+'. Omitted this parameter disables "
                 "checkpointing."
        )
        self.add_argument(
            '--checkpointing-best-only',
            action='store_true',
            default=False,
            help="Store only the best checkpoint."
        )
        self.add_argument(
            '--checkpointing-skip',
            type=float,
            default=0.0,
            help="Skip checkpointing in early epochs. For example, passing a"
                 "value of '0.2' and `n_epochs` of '500', skips checkpointing "
                 "for the first 0.2*500 = 100 epochs. A value of '1.0' "
                 "disables checkpointing at all."
        )

        # debugging ------------------------------------------------------------
        self.add_argument(
            '--debug',
            action='store_true',
            default=False,
            help="Enables debug outputs."
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
        # Weights & Biases -----------------------------------------------------
        self.add_argument(
            '--wandb-mode',
            type=str,
            choices=('online', 'offline', 'disabled'),     # see wandb
            default='online',
            help="Mode for Weights & Biases"
        )
        self.add_argument(
            '--wandb-name',
            type=str,
            default='EMSANet',
            help="Project name for Weights & Biases"
        )

        # other parameters -----------------------------------------------------
        self.add_argument(
            '--notes',
            type=str,
            default='',
            help="Just to add some additional notes for this run."
        )

    def parse_args(self, args=None, namespace=None, verbose=True):
        # parse args
        pa = super(ArgParserEMSANet, self).parse_args(
            args=args,
            namespace=namespace
        )

        def _warn(text):
            if verbose:
                warnings.warn(text)

        # convert nargs+ arguments from lists to tuples ------------------------
        pa.input_modalities = tuple(pa.input_modalities)
        pa.encoder_decoder_skip_downsamplings = tuple(
            pa.encoder_decoder_skip_downsamplings
        )
        pa.tasks = tuple(pa.tasks)
        if pa.checkpointing_metrics is not None:
            pa.checkpointing_metrics = tuple(pa.checkpointing_metrics)
        if pa.tasks_weighting is not None:
            pa.tasks_weighting = tuple(pa.tasks_weighting)
        pa.instance_weighting = tuple(pa.instance_weighting)

        # perform some initial argument checks
        # weights filepaths ----------------------------------------------------
        if pa.encoder_backbone_pretrained_weights_filepath is not None:
            # check if filepaths for rgb and depth are not set
            if any((pa.rgb_encoder_backbone_pretrained_weights_filepath is not None,
                    pa.depth_encoder_backbone_pretrained_weights_filepath is not None)):
                raise ValueError(
                    "Only use `encoder-backbone-pretrained-weights-filepath` "
                    "if you want to initialize both encoder backbones with the "
                    "same weights! "
                    "`rgb-encoder-backbone-pretrained-weights-filepath` and "
                    "`depth-encoder-backbone-pretrained-weights-filepath` must "
                    "not be set."
                )
            pa.rgb_encoder_backbone_pretrained_weights_filepath = \
                pa.encoder_backbone_pretrained_weights_filepath
            pa.depth_encoder_backbone_pretrained_weights_filepath = \
                pa.encoder_backbone_pretrained_weights_filepath
        # this argument is not needed anymore
        del pa.encoder_backbone_pretrained_weights_filepath

        # model ----------------------------------------------------------------
        if 1 == len(pa.input_modalities):
            pa.encoder_fusion = 'none'
            _warn("Set `encoder-fusion` to 'none' as there is only one input "
                  "modality.")
            if pa.input_modalities[0] == 'depth':
                if 'add-rgb' == pa.encoder_decoder_fusion:
                    pa.encoder_decoder_fusion = 'add-depth'
                    _warn("Changed `encoder-decoder-fusion` from 'add-rgb' "
                          "to 'add-depth' as `input-modalities` is 'depth'.")

        # multi-task parameters ------------------------------------------------
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
        # training -------------------------------------------------------------
        if pa.batch_size != 8:
            # the provided learning rate refers to the default batch size of 8
            # when using different batch sizes, we need to adjust the learning
            # rate accordingly
            pa.learning_rate = pa.learning_rate * pa.batch_size / 8
            _warn(f"Adapting learning rate to '{pa.learning_rate}' as the "
                  f"provided batch size differs from default batch size of 8.")

        if pa.tasks_weighting is None:
            # default weighting (required for inference or timing)
            pa.tasks_weighting = (1,)*len(pa.tasks)

        if len(pa.tasks_weighting) != len(pa.tasks):
            raise ValueError("Length for given task weighting does not match "
                             f"number of tasks: {len(pa.tasks_weighting)} vs. "
                             f"{len(pa.tasks)}.")

        if pa.dataset == 'coco':
            if 'depth' in pa.input_modalities:
                raise ValueError("COCO dataset does not feature depth data.")
            if 'normal' in pa.tasks:
                raise ValueError("COCO dataset does not feature surface "
                                 "normals.")
            if 'scene' in pa.tasks:
                raise ValueError("Scene classification is not supported for "
                                 "COCO dataset.")

        if pa.dataset in ('hypersim', 'cityscapes'):
            # Depth data for hypersim is clipped to the limit of png16 (uint16)
            # during dataset preparation. To account for that and to ignore
            # these pixels '--raw-depth' should be forced. Note, the actual
            # amount of clipped pixels is quite small.
            _warn(f"Forcing `raw-depth` as `dataset` is '{pa.dataset}'.")
            pa.raw_depth = True

        # evaluation -----------------------------------------------------------
        if pa.validation_full_resolution:
            if pa.dataset not in ('cityscapes', 'hypersim'):
                # height/width in cityscapes and hypersim are multiple 32
                raise ValueError(
                    "Validation with full resolution inputs is only supported"
                    "if `dataset` is 'cityscapes' or 'hypersim'."
                )
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
                    pa.weights_filepath)
                pa.visualization_output_path = os.path.join(
                    os.path.dirname(weights_dirpath),
                    f'visualization_{os.path.splitext(weights_filename)[0]}'
                )
            if os.path.exists(pa.visualization_output_path):
                raise ValueError(
                    "The path provided by `visualization-output-path` "
                    f"{pa.visualization_output_path} already exists. Please "
                    "provide a different path."
                )

        # TODO: can be removed from codebase later
        if all((pa.validation_only,
                pa.dataset == 'hypersim',
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

        # other parameters -----------------------------------------------------
        if pa.debug:
            _warn("`debug` is set, forcing 10 batches for training and "
                  "validation.")

        # print args
        if verbose:
            args_str = json.dumps(vars(pa), indent=4, sort_keys=True)
            print(f"Running with args:\n {args_str}")

        # return parsed (and subsequently modified) args
        return pa
