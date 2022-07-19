#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
Modified Panoptic-DeepLab Training Script.
The Code is based on:
    https://github.com/facebookresearch/detectron2/blob/main/projects/Panoptic-DeepLab/train_net.py
"""

import os
import torch

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    DatasetEvaluators,
)
from detectron2.projects.deeplab import build_lr_scheduler
from detectron2.projects.panoptic_deeplab import (
    add_panoptic_deeplab_config,
)
from detectron2.solver import get_default_optimizer_params
from detectron2.solver.build import maybe_add_gradient_clipping

from detectron2.config import CfgNode

from nicr_scene_analysis_datasets import d2 as nicr_d2
from dataset_mapper import PanopticDeeplabDatasetDictMapper
from dataset_mapper import DatasetMapperDict
import math
import json

from PIL import Image

from evaluator import COCOPanopticEvaluatorMod
from utils import create_coco_pan_gt


def build_sem_seg_train_aug(cfg):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
    augs.append(T.RandomFlip())
    return augs


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.epoch_counter = 1
        # Ugly hack to set the epoch in the evaluation.
        os.environ["EPOCH_COUNTER"] = str(self.epoch_counter)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if cfg.MODEL.PANOPTIC_DEEPLAB.BENCHMARK_NETWORK_SPEED:
            return None
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco_panoptic_seg":
            # When using the nicr dataset, we don't have the required files
            # for the panoptic evaluator (panopticapi is file based).
            # The panoptic evaluator requires a json file which contains
            # metadata about the panoptic segmentation.
            # Furthermore, the panoptic evaluator requires the images to be
            # stored in a file.
            # Because we don't want the files to get created every time we
            # evalate, we create them here.
            # Creating them multiple times isn't necessary, because they
            # don't change for the ground truth.
            output_folder_gt = os.path.join(output_folder, dataset_name,
                                            "panoptic_gt")
            output_pan_gt_json_file = os.path.join(output_folder_gt,
                                                   'panoptic_gt.json')
            MetadataCatalog.get(dataset_name).set(panoptic_root=output_folder_gt,
                                                  panoptic_json=output_pan_gt_json_file)

            if not os.path.isdir(output_folder_gt):
                create_coco_pan_gt(dataset_name,
                                   output_folder_gt,
                                   output_pan_gt_json_file)
            output_folder_pred = os.path.join(output_folder, dataset_name,
                                              "panoptic_pred")
            evaluator_list.append(
                COCOPanopticEvaluatorMod(dataset_name,
                                         output_folder_pred,
                                         # Ugly hack to get the current epoch.
                                         os.environ.get('EPOCH_COUNTER'),
                                         cfg.EVALUATOR.SAVE_EVAL_IMAGES))
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "No Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # The train load is a modified a little bit, as we want to use the
        # NICR Scene Analysis Datasets.
        # First we load the correct dataset config, which then gets used, by
        # the NICR DatasetMapper, so all required keys are in the dict.
        # E.g. it's add the pan_seg key.
        # After that the PanopticDeeplabMapper code is used for adding the
        # offset vectors and the heatmap.
        # For chaining those two mappers, the NICRChainedDatasetMapper is used.
        dataset_config = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).dataset_config
        nicr_mapper = nicr_d2.NICRSceneAnalysisDatasetMapper(dataset_config)
        panoptic_mapper = PanopticDeeplabDatasetDictMapper(
            cfg,
            augmentations=build_sem_seg_train_aug(cfg)
        )
        mapper = nicr_d2.NICRChainedDatasetMapper(
            [nicr_mapper, panoptic_mapper]
        )
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        # Does pretty much the same as build_train_loader
        dataset = DatasetCatalog.get(dataset_name)
        dataset_config = MetadataCatalog.get(dataset_name).dataset_config
        nicr_mapper = nicr_d2.NICRSceneAnalysisDatasetMapper(dataset_config)
        test_mapper = DatasetMapperDict(cfg, False)
        mapper = nicr_d2.NICRChainedDatasetMapper(
            [nicr_mapper, test_mapper]
        )
        return build_detection_test_loader(dataset, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Build an optimizer from config.
        """
        params = get_default_optimizer_params(
            model,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
        )

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
                params,
                cfg.SOLVER.BASE_LR,
                momentum=cfg.SOLVER.MOMENTUM,
                nesterov=cfg.SOLVER.NESTEROV,
            )
        elif optimizer_type == "ADAM":
            return maybe_add_gradient_clipping(cfg, torch.optim.Adam)(params, cfg.SOLVER.BASE_LR)
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")

    def test(self, cfg, model, evaluators=None):
        # We only want to evalaute every n epochs in early epochs, as evaluation
        # takes quiet some time and slows down training a lot.
        # At the end of the training we evlauate every epoch, because we don't
        # wan't to miss a good checkpoint.
        if self.epoch_counter < cfg.EVALUATOR.EVALUATE_EVERY_EPOCH_STARTING_AT:
            if self.epoch_counter % cfg.EVALUATOR.EVALUATE_EVERY_N_EPOCHS:
                r = {'epoch': self.epoch_counter}
                self.epoch_counter += 1
                os.environ["EPOCH_COUNTER"] = str(self.epoch_counter)
                return r
        r = super().test(cfg, model, evaluators)
        r["epoch"] = self.epoch_counter
        self.epoch_counter += 1
        # This is a ugly hack, which is used to get the current epoch count in
        # the evaluators.
        # Sometimes we wan't to safe the panoptic images.
        # In this case, the epoch_counter is used to create a folder for the
        # related epoch.
        os.environ["EPOCH_COUNTER"] = str(self.epoch_counter)
        return r


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_panoptic_deeplab_config(cfg)
    # Add custom keys
    cfg.DATASETS.BASE_PATH = 'datasets'
    cfg.SOLVER.EPOCHS = 500
    cfg.EVALUATOR = CfgNode()
    cfg.EVALUATOR.EVALUATE_EVERY_N_EPOCHS = 20
    cfg.EVALUATOR.EVALUATE_EVERY_EPOCH_STARTING_AT = 450
    cfg.EVALUATOR.SAVE_EVAL_IMAGES = False

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Setup the dataset for nicr_scene_analysis_datasets
    # e.g. cfg.DATASETS.TRAIN -> ("nyuv2_train",)
    dataset_base_name = cfg.DATASETS.TRAIN[0].split('_')[0]
    nicr_dataset_name = f'nicr-scene-analysis-datasets-{dataset_base_name}-v030'
    dataset_path = cfg.DATASETS.BASE_PATH
    nicr_d2.set_dataset_path(dataset_path)

    # Change config so we can train in epochs
    epochs = cfg.SOLVER.EPOCHS
    # Get number of samples in training dataset
    num_train = len(DatasetCatalog.get(cfg.DATASETS.TRAIN[0]))
    batch_size = cfg.SOLVER.IMS_PER_BATCH

    cfg.SOLVER.MAX_ITER = math.ceil((num_train * epochs)/batch_size)
    cfg.TEST.EVAL_PERIOD = math.ceil(num_train/batch_size * 1)
    cfg.SOLVER.CHECKPOINT_PERIOD = cfg.TEST.EVAL_PERIOD
    cfg.SOLVER.WARMUP_ITERS = math.ceil(num_train/batch_size * 50)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
