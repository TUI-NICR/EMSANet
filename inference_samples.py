# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from glob import glob
import os

import cv2
import matplotlib.pyplot as plt
import torch

from nicr_mt_scene_analysis.data import move_batch_to_device
from nicr_mt_scene_analysis.data import mt_collate

from emsanet.args import ArgParserEMSANet
from emsanet.data import get_datahelper
from emsanet.model import EMSANet
from emsanet.preprocessing import get_preprocessor
from emsanet.visualization import visualize_predictions
from emsanet.weights import load_weights


def _get_args():
    parser = ArgParserEMSANet()

    # add additional arguments
    group = parser.add_argument_group('Inference')
    group.add_argument(    # useful for appm context module
        '--inference-input-height',
        type=int,
        default=480,
        dest='validation_input_height',    # used in test phase
        help="Network input height for predicting on inference data."
    )
    group.add_argument(    # useful for appm context module
        '--inference-input-width',
        type=int,
        default=640,
        dest='validation_input_width',    # used in test phase
        help="Network input width for predicting on inference data."
    )
    group.add_argument(
        '--depth-max',
        type=float,
        default=None,
        help="Additional max depth values. Values above are set to zero as "
             "they are most likely not valid. Note, this clipping is applied "
             "before scaling the depth values."
    )
    group.add_argument(
        '--depth-scale',
        type=float,
        default=1.0,
        help="Additional depth scaling factor to apply."
    )

    return parser.parse_args()


def _load_img(fp):
    img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def main():
    args = _get_args()
    assert all(x in args.input_modalities for x in ('rgb', 'depth')), \
        "Only RGBD inference supported so far"

    device = torch.device('cuda')

    # data and model
    data = get_datahelper(args)
    dataset_config = data.dataset_config
    model = EMSANet(args, dataset_config=dataset_config)

    # load weights
    print(f"Loading checkpoint: '{args.weights_filepath}'")
    checkpoint = torch.load(args.weights_filepath)
    state_dict = checkpoint['state_dict']
    if 'epoch' in checkpoint:
        print(f"-> Epoch: {checkpoint['epoch']}")
    load_weights(args, model, state_dict, verbose=True)

    torch.set_grad_enabled(False)
    model.eval()
    model.to(device)

    # build preprocessor
    preprocessor = get_preprocessor(
        args,
        dataset=data.datasets_valid[0],
        phase='test',
        multiscale_downscales=None
    )

    # get samples
    basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'samples')
    rgb_filepaths = sorted(glob(os.path.join(basepath, '*_rgb.*')))
    depth_filepaths = sorted(glob(os.path.join(basepath, '*_depth.*')))
    assert len(rgb_filepaths) == len(depth_filepaths)

    for fp_rgb, fp_depth in zip(rgb_filepaths, depth_filepaths):
        # load rgb and depth image
        img_rgb = _load_img(fp_rgb)

        img_depth = _load_img(fp_depth).astype('float32')
        if args.depth_max is not None:
            img_depth[img_depth > args.depth_max] = 0
        img_depth *= args.depth_scale

        # preprocess sample
        sample = preprocessor({
            'rgb': img_rgb,
            'depth': img_depth,
            'identifier': os.path.basename(os.path.splitext(fp_rgb)[0])
        })

        # add batch axis as there is no dataloader
        batch = mt_collate([sample])
        batch = move_batch_to_device(batch, device=device)

        # apply model
        predictions = model(batch, do_postprocessing=True)

        # visualize predictions
        preds_viz = visualize_predictions(
            predictions=predictions,
            batch=batch,
            dataset_config=dataset_config
        )

        # show results
        _, axs = plt.subplots(2, 4, figsize=(12, 6), dpi=150)
        [ax.set_axis_off() for ax in axs.ravel()]

        axs[0, 0].set_title('RGB')
        axs[0, 0].imshow(
            img_rgb
        )
        axs[0, 1].set_title('Depth')
        axs[0, 1].imshow(
            img_depth,
            interpolation='nearest'
        )
        axs[0, 2].set_title('Semantic')
        axs[0, 2].imshow(
            preds_viz['semantic_segmentation_idx_fullres'][0],
            interpolation='nearest'
        )
        axs[0, 3].set_title('Semantic (panoptic)')
        axs[0, 3].imshow(
            preds_viz['panoptic_segmentation_deeplab_semantic_idx_fullres'][0],
            interpolation='nearest'
        )
        axs[1, 0].set_title('Instance (panoptic)')
        axs[1, 0].imshow(
            preds_viz['panoptic_segmentation_deeplab_instance_idx_fullres'][0],
            interpolation='nearest'
        )
        axs[1, 1].set_title('Instance centers')
        axs[1, 1].imshow(
            preds_viz['instance_centers'][0]
        )
        axs[1, 2].set_title('Instance offsets')
        axs[1, 2].imshow(
            preds_viz['instance_offsets'][0]
        )
        axs[1, 3].set_title('Panoptic (with orientations)')
        axs[1, 3].imshow(
            preds_viz['panoptic_orientations_fullres'][0],
            interpolation='nearest'
        )

        plt.suptitle(
            f"Image: ({os.path.basename(fp_rgb)}, "
            f"{os.path.basename(fp_depth)}), "
            f"Model: {args.weights_filepath}, "
            f"Scene: {preds_viz['scene'][0]}"
        )
        plt.tight_layout()

        # fp = os.path.join('./', 'samples', f'result_{args.dataset}.png')
        # plt.savefig(fp, bbox_inches='tight', pad_inches=0.05, dpi=150)

        plt.show()


if __name__ == '__main__':
    main()
