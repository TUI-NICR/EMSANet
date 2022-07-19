# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>

Some parts of this code are based on detectron2:
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/evaluation/panoptic_evaluation.py
"""

from builtins import super
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager

import itertools
import contextlib
import io
import os
import tempfile
import json
import numpy as np

import logging

from detectron2.evaluation import (
    COCOPanopticEvaluator,
    SemSegEvaluator
)

import torch

from detectron2.evaluation.panoptic_evaluation import _print_panoptic_results

from detectron2.data import DatasetCatalog, MetadataCatalog
import cv2


class COCOPanopticEvaluatorMod(COCOPanopticEvaluator):

    def __init__(self, dataset_name, output_dir,
                 epoch_counter, save_eval_images):
        # This class wraps the COCOPanopticEvaluator so we can save
        # some of the evaluation images.
        # Lots of code is copied from COCOPanopticEvaluator.
        super().__init__(dataset_name, output_dir)
        self.epoch_counter = int(epoch_counter)
        self.save_images = save_eval_images

    def process(self, inputs, outputs):
        super().process(inputs, outputs)
        if not self.save_images:
            # Nothing to do.
            return

        # Ugly hack to save the semantic images
        output_folder = os.path.join(self._output_dir,
                                     'semantic_pred',
                                     f'epoch_{self.epoch_counter}')
        os.makedirs(output_folder, exist_ok=True)
        for input, output in zip(inputs, outputs):
            sem_seg = output['sem_seg']
            sem_seg = sem_seg.argmax(dim=0).cpu().numpy()
            output_file = os.path.join(output_folder, input['file_name'])
            cv2.imwrite(output_file, sem_seg)

    def evaluate(self):
        comm.synchronize()

        self._predictions = comm.gather(self._predictions)
        self._predictions = list(itertools.chain(*self._predictions))
        if not comm.is_main_process():
            return

        # PanopticApi requires local files
        gt_json = PathManager.get_local_path(self._metadata.panoptic_json)
        gt_folder = PathManager.get_local_path(self._metadata.panoptic_root)
        # This dir is used when the evalutation images get saved.
        # Else a temp dir is used, which gets deleted after the evaluation.
        pred_dir = self._output_dir
        with tempfile.TemporaryDirectory(prefix="panoptic_eval") as pred_dir_tmp:
            for p in self._predictions:
                t_output_dir = None
                if self.save_images:
                    t_output_dir = pred_dir
                else:
                    t_output_dir = pred_dir_tmp
                with open(os.path.join(t_output_dir, p["file_name"]), "wb") as f:
                    f.write(p.pop("png_string"))

            with open(gt_json, "r") as f:
                json_data = json.load(f)
            json_data["annotations"] = self._predictions

            output_dir = self._output_dir or pred_dir
            predictions_json = os.path.join(output_dir, "predictions.json")
            with PathManager.open(predictions_json, "w") as f:
                f.write(json.dumps(json_data))

            from panopticapi.evaluation import pq_compute

            with contextlib.redirect_stdout(io.StringIO()):
                pq_res = pq_compute(
                    gt_json,
                    PathManager.get_local_path(predictions_json),
                    gt_folder=gt_folder,
                    pred_folder=pred_dir,
                )

        res = {}
        res["PQ"] = 100 * pq_res["All"]["pq"]
        res["SQ"] = 100 * pq_res["All"]["sq"]
        res["RQ"] = 100 * pq_res["All"]["rq"]
        res["PQ_th"] = 100 * pq_res["Things"]["pq"]
        res["SQ_th"] = 100 * pq_res["Things"]["sq"]
        res["RQ_th"] = 100 * pq_res["Things"]["rq"]
        res["PQ_st"] = 100 * pq_res["Stuff"]["pq"]
        res["SQ_st"] = 100 * pq_res["Stuff"]["sq"]
        res["RQ_st"] = 100 * pq_res["Stuff"]["rq"]

        results = dict({"panoptic_seg": res})
        _print_panoptic_results(pq_res)

        return results
