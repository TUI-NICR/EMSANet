# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
"""
from nicr_mt_scene_analysis.testing.onnx import export_onnx_model
import onnx
import torch

from emsanet.args import ArgParserEMSANet
from emsanet.data import get_datahelper
from emsanet.model import EMSANet


def test_weights():
    """
    test that all weights are part of the state dict and exported correctly
    to onnx
    """
    # args and data needed for building a model
    parser = ArgParserEMSANet()
    args = parser.parse_args('', verbose=False)
    args.no_pretrained_backbone = True
    args.dropout_p = 0
    args.validation_batch_size = 2

    data = get_datahelper(args)
    dataset_config = data.dataset_config

    # build model and extract weights
    model_1 = EMSANet(args, dataset_config=dataset_config)
    model_1.eval()
    state_dict_1 = model_1.state_dict()

    # build second model and load weights from first model
    model_2 = EMSANet(args, dataset_config=dataset_config)
    model_2.load_state_dict(state_dict_1)
    model_2.eval()

    # prepare input for onnx export
    batch_size = 3
    input_shape = (480, 640)
    batch = {}
    if 'rgb' in args.input_modalities:
        batch['rgb'] = torch.randn((batch_size, 3)+input_shape)
    if 'depth' in args.input_modalities:
        batch['depth'] = torch.randn((batch_size, 1)+input_shape)

    x = (batch, {'do_postprocessing': False})

    # export both models to onnx
    export_onnx_model('model_1.onnx', model_1, x, force_export=True)
    export_onnx_model('model_2.onnx', model_2, x, force_export=True)

    # load onnx models again
    onnx_model_1 = onnx.load('model_1.onnx')
    onnx_model_2 = onnx.load('model_2.onnx')

    # test if weights for each layer are the same between first and second model
    for l1, l2 in zip(onnx_model_1.graph.initializer,
                      onnx_model_2.graph.initializer):
        if l1.raw_data != l2.raw_data:
            raise
