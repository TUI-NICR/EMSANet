#!/bin/bash
set -o xtrace

ARGS_DEFAULT='--dataset nyuv2 --dataset-path ./datasets/nyuv2 --weights-filepath ./trained_models/nyuv2/r34_NBt1D_pre.pth'

ARGS_RESNET18='--rgb-encoder-backbone resnet18 --depth-encoder-backbone resnet18'
ARGS_RESNET34='--rgb-encoder-backbone resnet34 --depth-encoder-backbone resnet34'
ARGS_RESNET50='--rgb-encoder-backbone resnet50 --depth-encoder-backbone resnet50'
ARGS_RESNET101='--rgb-encoder-backbone resnet101 --depth-encoder-backbone resnet101'

ARGS_BASICBLOCK='--rgb-encoder-backbone-block basicblock --depth-encoder-backbone-block basicblock'
ARGS_NONBOTTLENECK1D='--rgb-encoder-backbone-block nonbottleneck1d --depth-encoder-backbone-block nonbottleneck1d'
ARGS_BOTTLENECK='--rgb-encoder-backbone-block bottleneck --depth-encoder-backbone-block bottleneck'

ARGS_TIME_PYTORCH='--n-runs-warmup 20 --n-runs 80 --no-time-tensorrt'
ARGS_EXPORT_ONNX_TRT='--no-time-pytorch --trt-onnx-export-only --trt-onnx-opset-version 11'
ARGS_TIME_TRT32='--model-onnx-filepath ./model_tensorrt.onnx --n-runs-warmup 20 --n-runs 80 --no-time-pytorch'
ARGS_TIME_TRT16='--model-onnx-filepath ./model_tensorrt.onnx --n-runs-warmup 20 --n-runs 80 --no-time-pytorch --trt-floatx 16'

SED_PYTORCH="sed -n 's/.*fps pytorch: \([0-9.]*\) ± \([0-9.]*\).*$/\1 \2,/p'"
SED_TRT32="sed -n 's/.*fps tensorrt: \([0-9.]*\) ± \([0-9.]*\).*$/\1 \2,/p'"
SED_TRT16="sed -n 's/.*fps tensorrt: \([0-9.]*\) ± \([0-9.]*\).*$/\1 \2,/p'"

# ------------------------------------------------------------------------------
RESULTS_FILE='./nyuv2_timings_jetson_all.csv'

# multi task
for TASKS in '--tasks semantic instance orientation scene --enable-panoptic'
do
    for ARGS_MODALITY in '--input-modalities rgb depth'
    do
        # resnet34 nonbottleneck1d
        ARGS="${ARGS_DEFAULT} ${TASKS} ${ARGS_MODALITY} ${ARGS_RESNET34} ${ARGS_NONBOTTLENECK1D}"
        echo -n "${ARGS}," >> $RESULTS_FILE
        python3 inference_time_whole_model.py $ARGS_TIME_PYTORCH $ARGS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE

        # export onnx model first and time in second call -> saves resources
        python3 inference_time_whole_model.py $ARGS_EXPORT_ONNX_TRT $ARGS
        python3 inference_time_whole_model.py $ARGS_TIME_TRT32 $ARGS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
        python3 inference_time_whole_model.py $ARGS_TIME_TRT16 $ARGS | eval $SED_TRT16 >> $RESULTS_FILE
    done
done
