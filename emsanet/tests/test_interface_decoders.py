# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
import os

import pytest
import torch

from nicr_mt_scene_analysis.testing.onnx import export_onnx_model

from emsanet.args import ArgParserEMSANet
from emsanet.decoder import get_decoders


class ForwardHelper(torch.nn.Module):
    def __init__(self, decoders):
        super().__init__()
        self.decoders = decoders

    def forward(self, x, skips, batch, do_postprocessing=False):
        outs = []
        for decoder in self.decoders.values():
            outs.append(
                decoder(x, skips, batch,
                        do_postprocessing=do_postprocessing)
            )

        return outs


def decoders_test(args, do_postprocessing, training, tmp_path):
    # create decoders
    debug = True

    decoders = get_decoders(
        args,
        n_channels_in=512,
        downsampling_in=32,
        semantic_n_blocks=3,
        instance_n_blocks=2,
        normal_n_blocks=1,
        scene_n_channels_in=512//2,
        fusion_n_channels=(256, 128, 64),
        debug=debug
    )

    input_h, input_w = (480, 640)
    downsampling_in = 32

    n_decoders = len(args.tasks)

    if 'orientation' in args.tasks:
        # orientation task is handled in instance decoder
        n_decoders -= 1

    if args.enable_panoptic:
        # panoptic task fuses semantic and instance
        n_decoders -= 1

    assert len(decoders) == n_decoders

    # create model containing all decoders
    model = ForwardHelper(decoders)
    if not training:
        model.eval()

    # set up inputs for decoders
    x = (
        # output of context module
        torch.rand(3, 512, input_h//downsampling_in, input_w//downsampling_in),
        # at least one context branch (from GAP)
        (torch.rand(3, 512//2, 1, 1),)
    )
    # strings are used to prevent casting keys from int to tensor(int)
    # while exporting to ONNX
    skips = {
            '16': {'rgb': torch.rand(3, 256, input_h//16, input_w//16),
                   'depth': torch.rand(3, 256, input_h//16, input_w//16)},
            '8': {'rgb': torch.rand(3, 128, input_h//8, input_w//8),
                  'depth': torch.rand(3, 128, input_h//8, input_w//8)},
            '4': {'rgb': torch.rand(3, 64, input_h//4, input_w//4),
                  'depth': torch.rand(3, 64, input_h//4, input_w//4)},
    }
    batch = {}
    if 'instance' in args.tasks:
        # pure instance segmentation task requires gt foreground mask
        batch['instance_foreground'] = torch.ones((3, input_h, input_w),
                                                  dtype=torch.bool)
    if 'orientation' in args.tasks:
        # orientation estimation requires a gt segmentation and foreground mask
        batch['instance'] = torch.ones((3, input_h, input_w), dtype=torch.bool)
        batch['orientation_foreground'] = torch.ones((3, input_h, input_w),
                                                     dtype=torch.bool)

    if not training and do_postprocessing:
        # for inference postprocessing, inputs in full resolution are required
        batch['rgb_fullres'] = torch.randn((3, 3, input_h, input_w))
        batch['depth_fullres'] = torch.randn((3, 1, input_h, input_w))

    # apply decoders
    outputs = model(x, skips, batch,
                    do_postprocessing=do_postprocessing)

    # perform some basic checks
    assert len(outputs) == n_decoders

    if not do_postprocessing:
        # output of decoder(s) is returned: tuple(outputs, side_outputs)
        for output in outputs:
            assert isinstance(output, tuple)
            assert len(output) == 2
    else:
        # postprocessed output of decoders is returned: dict
        for output in outputs:
            assert isinstance(output, dict)
            assert len(output)

    # export decoders to ONNX
    if not training and do_postprocessing:
        # stop here: inference postprocessing is challenging
        return
    # determine filename and filepath
    tasks_str = '+'.join(args.tasks)
    if args.enable_panoptic:
        tasks_str += '+panoptic'
    filename = f'decoders_{tasks_str}'
    filename += f'__train_{training}'
    filename += f'__post_{do_postprocessing}'
    filename += '.onnx'
    filepath = os.path.join(tmp_path, filename)
    # export
    x = (x, skips, batch, {'do_postprocessing': do_postprocessing})
    export_onnx_model(filepath, model, x)


@pytest.mark.parametrize('enable_panoptic', (False, True))
@pytest.mark.parametrize('do_postprocessing', (False, True))
@pytest.mark.parametrize('training', (False, True))
def test_decoders_full_mt_emsanet(enable_panoptic, do_postprocessing, training,
                                  tmp_path):
    """Test EMSANet decoders in full mt setting"""
    parser = ArgParserEMSANet()
    args = parser.parse_args('', verbose=False)
    args.tasks = ('semantic',
                  'instance', 'orientation',
                  'normal',
                  'scene')
    args.enable_panoptic = enable_panoptic
    decoders_test(args,
                  do_postprocessing=do_postprocessing,
                  training=training,
                  tmp_path=tmp_path)


@pytest.mark.parametrize('enable_panoptic', (False, True))
@pytest.mark.parametrize('do_postprocessing', (False, True))
@pytest.mark.parametrize('training', (False, True))
def test_decoders_full_mt_segformermlp(enable_panoptic, do_postprocessing,
                                       training, tmp_path):
    """Test SegFormerMLP decoders in full mt setting"""
    parser = ArgParserEMSANet()
    args = parser.parse_args([
        # semantic
        '--semantic-decoder', 'segformermlp',
        '--semantic-decoder-n-channels', '256', '128', '64', '64',
        '--semantic-decoder-upsampling', 'bilinear',
        '--semantic-encoder-decoder-fusion', 'select-rgb',
        # instance
        '--instance-decoder', 'segformermlp',
        '--instance-decoder-n-channels', '256', '128', '64', '64',
        '--instance-decoder-upsampling', 'bilinear',
        '--instance-encoder-decoder-fusion', 'select-depth',  # test depth
        # normal
        '--normal-decoder', 'segformermlp',
        '--normal-decoder-n-channels', '256', '128', '64', '64',
        '--normal-decoder-upsampling', 'bilinear',
        '--normal-encoder-decoder-fusion', 'select-rgb',
    ], verbose=False)
    args.tasks = ('semantic',
                  'instance', 'orientation',
                  'normal',
                  'scene')
    args.enable_panoptic = enable_panoptic
    decoders_test(args,
                  do_postprocessing=do_postprocessing,
                  training=training,
                  tmp_path=tmp_path)


@pytest.mark.parametrize('do_postprocessing', (False, True))
@pytest.mark.parametrize('training', (False, True))
def test_decoders_panoptic_mixed(do_postprocessing, training, tmp_path):
    """Test decoders in panoptic setting with mixed decoder types"""
    parser = ArgParserEMSANet()
    args = parser.parse_args([
        # semantic
        '--semantic-decoder', 'segformermlp',
        '--semantic-decoder-n-channels', '256', '128', '64', '64',
        '--semantic-decoder-upsampling', 'bilinear',
        '--semantic-encoder-decoder-fusion', 'select-depth',  # test depth
        # instance (= default args)
        '--instance-decoder', 'emsanet',
        '--instance-decoder-n-channels', '512', '256', '128',
        '--instance-decoder-upsampling', 'learned-3x3-zeropad',
        '--instance-encoder-decoder-fusion', 'add-rgb',
        '--tasks', 'semantic', 'instance',
        '--enable-panoptic',
    ], verbose=False)
    decoders_test(args,
                  do_postprocessing=do_postprocessing,
                  training=training,
                  tmp_path=tmp_path)
