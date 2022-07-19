# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

Notes:
- matching inputs/outputs of the onnx model to pass them to the
  postprocessors is not quite stable (just a fast proof-of-concept
  implementation)
- postprocessing is always done using PyTorch (on GPU if available) and not
  much optimized so far (many operations could be done using ONNX) and, thus,
  should not be part of a timing comparison
"""
from typing import Tuple

import os
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass    # pip3 install dataclasses (backport for python 3.6)

import matplotlib.pyplot as plt
from nicr_mt_scene_analysis.data import move_batch_to_device
import numpy as np
import torch

from emsanet.model import EMSANet
from emsanet.args import ArgParserEMSANet
from emsanet.data import get_datahelper
from emsanet.data import get_dataset
from emsanet.preprocessing import get_preprocessor
from emsanet.visualization import visualize


def _parse_args():
    parser = ArgParserEMSANet()

    # add arguments
    # general
    parser.add_argument(
        '--model-onnx-filepath',
        type=str,
        default=None,
        help="Path to ONNX model file when `model` is 'onnx'."
    )

    # runs
    parser.add_argument(
        '--n-runs',
        type=int,
        default=100,
        help="Number of runs the inference time will be measured."
    )
    parser.add_argument(
        '--n-runs-warmup',
        type=int,
        default=10,
        help="Number of forward passes through the model before the inference "
             "time measurements starts. This is necessary as the first runs "
             "are slower."
    )
    # timings
    parser.add_argument(
        '--no-time-pytorch',
        action='store_true',
        default=False,
        help="Do not measure inference time using PyTorch."
    )
    parser.add_argument(
        '--no-time-tensorrt',
        action='store_true',
        default=False,
        help="Do not measure inference time using TensorRT."
    )
    parser.add_argument(
        '--with-postprocessing',
        action='store_true',
        default=False,
        help="Include postprocessing in timing."
    )

    # plots / export
    parser.add_argument(
        '--plot-timing',
        action='store_true',
        default=False,
        help="Whether to plot the inference times for each forward pass."
    )
    parser.add_argument(
        '--export-outputs',
        action='store_true',
        default=False,
        help="Whether to export the outputs of the model."
    )

    # tensorrt
    parser.add_argument(
        '--trt-workspace',
        type=int,
        default=2 << 30,
        help="Maximum workspace size, default equals 2GB."
    )
    parser.add_argument(
        '--trt-floatx',
        type=int,
        choices=(16, 32),
        default=32,
        help="Whether to measure with float16 or float32."
    )
    parser.add_argument(
        '--trt-batchsize',
        type=int,
        default=1,
        help="Batchsize to use."
    )
    parser.add_argument(
        '--trt-onnx-opset-version',
        type=int,
        default=10,
        help="Opset version to use for export."
    )
    parser.add_argument(
        '--trt-do-not-force-rebuild',
        dest='trt_force_rebuild',
        action='store_false',
        default=True,
        help="Reuse existing TensorRT engine."
    )
    parser.add_argument(
        '--trt-enable-dynamic-batch-axis',
        action='store_true',
        default=False,
        help="Enable dynamic axes."
    )
    parser.add_argument(
        '--trt-onnx-export-only',
        action='store_true',
        default=False,
        help="Export ONNX model for TensorRT only. To measure inference time, "
             "use '--model-onnx-filepath ./model_tensorrt.onnx' in a second "
             "run."
    )

    args = parser.parse_args()
    return args


def get_engine(onnx_filepath,
               engine_filepath,
               trt_floatx=16,
               trt_batchsize=1,
               trt_workspace=2 << 30,
               force_rebuild=True):
    # note that we use onnx2trt from TensorRT Open Source Software Components
    # to convert ONNX files to TensorRT engines
    if not os.path.exists(engine_filepath) or force_rebuild:
        print("Building engine using onnx2trt")
        if trt_floatx == 32:
            print("... this may take a while")
        else:
            print("... this may take -> AGES <-")
        cmd = f'onnx2trt {onnx_filepath}'
        cmd += f' -d {trt_floatx}'    # 16: float16, 32: float32
        cmd += f' -b {trt_batchsize}'    # batchsize
        # cmd += ' -v'    # verbose
        # cmd += ' -l'    # list layers
        cmd += f' -w {trt_workspace}'   # workspace size mb
        cmd += f' -o {engine_filepath}'

        try:
            print(cmd)
            out = subprocess.check_output(cmd,
                                          shell=True,
                                          stderr=subprocess.STDOUT,
                                          universal_newlines=True)
        except subprocess.CalledProcessError as e:
            print("onnx2trt failed:", e.returncode, e.output)
            raise
        print(out)

    print(f"Loading engine: {engine_filepath}")
    with open(engine_filepath, "rb") as f, \
            trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def alloc_buf(engine):
    @dataclass
    class Binding:
        name: str
        shape: Tuple[int]
        cpu: np.array
        gpu: pycuda.driver.DeviceAllocation

    inputs, outputs = [], []
    for i in range(engine.num_bindings):
        # get name, shape, and dtype for binding
        name = engine.get_binding_name(i)
        shape = engine.get_binding_shape(i)
        trt_dtype = trt.nptype(engine.get_binding_dtype(i))

        # allocate memory
        host_mem = cuda.pagelocked_empty(trt.volume(shape), trt_dtype)
        dev_mem = cuda.mem_alloc(host_mem.nbytes)

        # create binding
        binding = Binding(name, tuple(shape), host_mem, dev_mem)

        # add to input output list
        if engine.binding_is_input(i):
            inputs.append(binding)
        else:
            outputs.append(binding)

    stream = cuda.Stream()
    return inputs, outputs, stream


def time_inference_pytorch(model,
                           inputs,
                           device,
                           n_runs_warmup=5,
                           with_postprocessing=False,
                           store_outputs=False):
    timings = []
    with torch.no_grad():
        outs = []
        for i, input_ in enumerate(inputs):
            # use PyTorch to time events
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            # copy to gpu
            inputs_gpu = {
                k: v.to(device)
                for k, v in input_.items()
                if ('rgb' in k or 'depth' in k) and torch.is_tensor(v)   # includes fullres
            }

            # model forward pass
            out_pytorch = model(inputs_gpu,
                                do_postprocessing=with_postprocessing)

            # copy back to cpu
            if not with_postprocessing:
                out_pytorch_cpu = []
                # output is tuple (outputs, side_output)
                for outputs, _ in out_pytorch:    # ignore side outputs
                    for output in outputs:
                        if isinstance(output, tuple):
                            # panoptic helper is again a tuple
                            out_pytorch_cpu.extend([o.cpu() for o in output])
                        else:
                            out_pytorch_cpu.append(output.cpu())
            else:
                # output is a dict
                out_pytorch_cpu = move_batch_to_device(out_pytorch, 'cpu')

            end.record()
            torch.cuda.synchronize()

            if i >= n_runs_warmup:
                timings.append(start.elapsed_time(end) / 1e3)

            if store_outputs:
                outs.append(out_pytorch_cpu)

    return np.array(timings), outs


def time_inference_tensorrt(onnx_filepath,
                            inputs,
                            trt_floatx=16,
                            trt_batchsize=1,
                            trt_workspace=2 << 30,
                            n_runs_warmup=5,
                            force_tensorrt_engine_rebuild=True,
                            postprocessors=None,
                            postprocessors_device='cpu',
                            store_outputs=False):
    # create engine
    trt_filepath = os.path.splitext(onnx_filepath)[0] + '.trt'

    engine = get_engine(onnx_filepath, trt_filepath,
                        trt_floatx=trt_floatx,
                        trt_batchsize=trt_batchsize,
                        trt_workspace=trt_workspace,
                        force_rebuild=force_tensorrt_engine_rebuild)
    context = engine.create_execution_context()

    # allocate memory on gpu
    input_bindings, output_bindings, stream = alloc_buf(engine)

    timings = []
    pointers = [int(in_.gpu) for in_ in input_bindings+output_bindings]
    in_keys = [b.name for b in input_bindings]
    out_keys = [b.name for b in output_bindings]

    outs = []
    for i, input_ in enumerate(inputs):
        start_time = time.time()

        # copy inputs to gpu
        for in_bind, key in zip(input_bindings, in_keys):
            cuda.memcpy_htod(in_bind.gpu, input_[key].numpy())

        # model forward pass
        context.execute(1, pointers)

        # copy back to cpu
        for out_bind in output_bindings:
            cuda.memcpy_dtoh(out_bind.cpu, out_bind.gpu)

        if postprocessors is None:
            # copy here slows down
            out_trt = [out_bind.cpu for out_bind in output_bindings]
        else:
            out_trt = {}
            for name, post in postprocessors.items():
                # create input
                # bit hacky, this works as the keys are ordered
                in_post = [
                    torch.from_numpy(out_bind.cpu).view(out_bind.shape)    # copy!
                    for out_bind in output_bindings
                    if name in out_bind.name
                ]

                if 'cpu' != postprocessors_device:
                    # copy back to GPU (not smart)
                    in_post = [t.to(postprocessors_device) for t in in_post]

                    # we also need some inputs on gpu for the postprocessing
                    input_gpu = {
                        k: v.to(device)
                        for k, v in input_.items()
                        if ('rgb' in k or 'depth' in k) and torch.is_tensor(v)   # includes fullres
                    }
                    input_ = input_gpu

                in_post_side = None
                if 1 == len(in_post):
                    # single input to postprocessor
                    in_post = in_post[0]
                else:
                    # multiple inputs to postprocessor (instance / panoptic)
                    in_post = tuple(in_post)

                if 'panoptic_helper' == name:
                    # this is not quite smart but works for now
                    # first element is semantic, the remaining instance
                    in_post = (in_post[0], in_post[1:])
                    in_post_side = None, None

                out_trt.update(
                    post.postprocess(data=(in_post, in_post_side),
                                     batch=input_,
                                     is_training=False)
                )

            # copy back to cpu
            if 'cpu' != postprocessors_device:
                out_trt = move_batch_to_device(out_trt, 'cpu')

        if i >= n_runs_warmup:
            timings.append(time.time() - start_time)

        if store_outputs:
            outs.append(out_trt)

    return np.array(timings), outs


if __name__ == '__main__':
    args = _parse_args()

    if args.trt_enable_dynamic_batch_axis and args.trt_onnx_opset_version < 11:
        warnings.warn("Dynamic batch axis requires opset 11 or higher.")

    print('PyTorch version:', torch.__version__)

    if not args.no_time_tensorrt:
        import tensorrt as trt
        import pycuda.autoinit     # pip3 install pycuda    # do not remove!
        import pycuda.driver as cuda

        print('TensorRT version:', trt.__version__)

    results_path = os.path.join(os.path.dirname(__file__),
                                f'inference_results',
                                args.dataset)
    os.makedirs(results_path, exist_ok=True)

    # prepare inputs -----------------------------------------------------------
    args.batch_size = 1
    args.validation_batch_size = 1
    args.n_workers = 0    # no threads in torch dataloaders, use main thread

    data_helper = get_datahelper(args)

    inputs = []
    if args.dataset_path is not None:
        # simply use first dataset (they all share the same properties)
        dataset = data_helper.datasets_valid[0]

        # get preprocessed samples of the given dataset
        data_helper.set_valid_preprocessor(
            get_preprocessor(
                args,
                dataset=dataset,
                phase='test',
                multiscale_downscales=None
            )
        )
        for sample in data_helper.valid_dataloaders[0]:
            inputs.append(sample)

            if (args.n_runs + args.n_runs_warmup) == len(inputs):
                # enough samples collected
                break
    else:
        dataset = get_dataset(args, split=args.validation_split)

        # we do not have a dataset, simply use random inputs
        if args.with_postprocessing:
            # postpressing random inputs does not really make sense
            # moreover, we need more fullres keys
            raise ValueError("Please provide a `dataset_path` to enable "
                             "inference with meaningful inputs.")

        # collect random inputs
        rgb_images = []
        depth_images = []
        for _ in range(args.n_runs + args.n_runs_warmup):
            img_rgb = np.random.randint(
                low=0,
                high=255,
                size=(args.input_height, args.input_width, 3),
                dtype='uint8'
            )
            img_depth = np.random.randint(
                low=0,
                high=40000,
                size=(args.input_height, args.input_width),
                dtype='uint16'
            )
            # preprocess
            img_rgb = (img_rgb / 255).astype('float32').transpose(2, 0, 1)
            img_depth = (img_depth.astype('float32') / 20000)[None]
            img_rgb = np.ascontiguousarray(img_rgb[None])
            img_depth = np.ascontiguousarray(img_depth[None])
            rgb_images.append(torch.tensor(img_rgb))
            depth_images.append(torch.tensor(img_depth))

        # convert to input format (see BatchType)
        if 2 == len(args.input_modalities):
            inputs = [{'rgb': rgb_images[i], 'depth': depth_images[i]}
                      for i in range(len(rgb_images))]
        elif 'rgb' in args.input_modalities:
            inputs = [{'rgb': rgb_images[i]}
                      for i in range(len(rgb_images))]
        elif 'depth' in args.input_modalities:
            inputs = [{'depth': depth_images[i]}
                      for i in range(len(rgb_images))]

    # create model ------------------------------------------------------------
    if args.model_onnx_filepath is not None:
        warnings.warn("PyTorch inference timing disabled since onnx model is "
                      "given.")
        args.no_time_pytorch = True

    # create model
    args.no_pretrained_backbone = True
    model = EMSANet(args=args, dataset_config=dataset.config)

    # load weights
    if args.weights_filepath is not None:
        checkpoint = torch.load(args.weights_filepath,
                                map_location=lambda storage, loc: storage)
        try:
            model.load_state_dict(checkpoint['state_dict'], strict=True)
        except RuntimeError as e:
            print("Load with strict=False due to", e)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
    device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
    model.eval()

    # time inference using PyTorch --------------------------------------------
    if not args.no_time_pytorch:
        # move model to gpu
        model.to(device)

        timings_pytorch, outs_pytorch = time_inference_pytorch(
            model,
            inputs,
            device,
            n_runs_warmup=args.n_runs_warmup,
            with_postprocessing=args.with_postprocessing,
            store_outputs=args.export_outputs
        )
        print(f'fps pytorch: {np.mean(1/timings_pytorch):0.4f} ± '
              f'{np.std(1/timings_pytorch):0.4f}')

        # move model back to cpu (required for further steps)
        model.to('cpu')

    # time inference using TensorRT -------------------------------------------
    if not args.no_time_tensorrt:
        if args.model_onnx_filepath is None:
            # we have to export the model to onnx

            # define dummy input for export
            dummy_input = ({k: v for k, v in inputs[0].items()
                            if k in args.input_modalities},
                           {'do_postprocessing': False})

            # define names for input and output graph nodes
            # note, meaningful names are required to match postprocessors and
            # to set up dynamic_axes dict correctly
            input_names = list(args.input_modalities)    # does it always work?

            # determine output structure in order to derive names
            outputs = model(dummy_input[0], **dummy_input[1])
            assert len(outputs) == len(model.decoders)
            # encode output structure to output names (note, this is parsed
            # later to assign the outputs to the postprocessors if the model
            # is loaded from pure onnx)
            output_names = []
            for (outs, side_outs), decoder_name in zip(outputs,
                                                       model.decoders):
                if not isinstance(outs, tuple):
                    # semantic (single tensor)
                    outs = tuple(outs)

                if 'panoptic_helper' == decoder_name:
                    # this is not quite smart but works for now
                    # join semantic (single tensor) and instance outputs
                    outs = (outs[0], ) + outs[1]

                for j, _ in enumerate(outs):
                    # format output name
                    output_names.append(f'{decoder_name}_{j}')

            # input_names = [f'input_{i}' for i in range(len(dummy_inputs))]
            # output_names = ['output']
            onnx_filepath = './model_tensorrt.onnx'

            if args.trt_enable_dynamic_batch_axis:
                kwargs = {
                    'dynamic_axes': {name: {0: "batch"}
                                     for name in input_names+output_names}
                }
            else:
                kwargs = {}

            torch.onnx.export(model,
                              dummy_input,
                              onnx_filepath,
                              export_params=True,
                              input_names=input_names,
                              output_names=output_names,
                              do_constant_folding=True,
                              verbose=False,
                              opset_version=args.trt_onnx_opset_version,
                              **kwargs)
            print(f"ONNX file (opset {args.trt_onnx_opset_version}) written "
                  f"to '{onnx_filepath}'.")

            if args.trt_onnx_export_only:
                # stop here
                sys.exit(0)
        else:
            onnx_filepath = args.model_onnx_filepath

        # extract postprocessors
        if args.with_postprocessing:
            postprocessors = {
                k: v.postprocessing for k, v in model.decoders.items()
            }
        else:
            postprocessors = None

        timings_tensorrt, outs_tensorrt = time_inference_tensorrt(
            onnx_filepath,
            inputs,
            trt_floatx=args.trt_floatx,
            trt_batchsize=args.trt_batchsize,
            trt_workspace=args.trt_workspace,
            n_runs_warmup=args.n_runs_warmup,
            force_tensorrt_engine_rebuild=args.trt_force_rebuild,
            postprocessors=postprocessors,
            postprocessors_device=device,
            store_outputs=args.export_outputs
        )

        print(f'fps tensorrt: {np.mean(1/timings_tensorrt):0.4f} ± '
              f'{np.std(1/timings_tensorrt):0.4f}')

    # plot/export results -----------------------------------------------------
    if args.plot_timing:
        plt.figure()
        if 'timings_pytorch' in locals():
            plt.plot(1 / timings_pytorch, label='pytorch')
        if 'timings_tensorrt' in locals():
            plt.plot(1 / timings_tensorrt, label='tensorrt')
        plt.xlabel("run")
        plt.ylabel("fps")
        plt.legend()
        plt.title("Inference time")
        plt.show()

    if args.export_outputs:
        assert args.with_postprocessing

        if 'outs_pytorch' in locals():
            for inp, out in zip(inputs, outs_pytorch):
                visualize(
                    output_path=os.path.join(results_path, 'pytorch'),
                    batch=inp,
                    predictions=out,
                    dataset_config=dataset.config
                )

        if 'outs_tensorrt' in locals():
            for inp, out in zip(inputs, outs_tensorrt):
                visualize(
                    output_path=os.path.join(results_path,
                                             f'tensorrt_{args.trt_floatx}'),
                    batch=inp,
                    predictions=out,
                    dataset_config=dataset.config
                )
