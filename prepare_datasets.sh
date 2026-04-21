#!/bin/bash
set -eo pipefail

prepare_datasets() {
    # NYUv2
    echo "Preparing NYUv2 dataset (~2 GB)"
    nicr_sa_prepare_dataset \
        nyuv2 \
        ./datasets/nyuv2

    # SUNRGB-D
    echo "Preparing SUNRGB-D dataset (~4 GB)"
    nicr_sa_prepare_dataset \
        sunrgbd \
        ./datasets/sunrgbd \
        --create-instances \
        --instances-version emsanet \
        --copy-instances-from-nyuv2 \
        --nyuv2-path ./datasets/nyuv2/
}

prepare_datasets_nyuv2_internal() {
    # NYUv2
    echo "Preparing NYUv2 dataset (~2 GB)"
    nicr_sa_prepare_dataset \
        nyuv2 \
        ./datasets/nyuv2 \
        --mat-filepath /datasets_nas/segmentation/nyuv2/nyu_depth_v2_labeled.mat \
        --enable-normal-extraction \
        --normal-filepath /datasets_nas/segmentation/nyuv2/normals_gt.tgz
}

prepare_datasets_sunrgbd_internal() {
    # SUNRGB-D
    echo "Preparing SUNRGB-D dataset (~4 GB)"
    nicr_sa_prepare_dataset \
        sunrgbd \
        ./datasets/sunrgbd \
        --create-instances \
        --instances-version emsanet \
        --copy-instances-from-nyuv2 \
        --nyuv2-path ./datasets/nyuv2/ \
        --toolbox-filepath /datasets_nas/segmentation/SunRGBD/SUNRGBDtoolbox.zip \
        --data-filepath /datasets_nas/segmentation/SunRGBD/SUNRGBD.zip \
        --box-filepath /datasets_nas/segmentation/SunRGBD/SUNRGBDMeta3DBB_v2.mat
}

prepare_datasets_hypersim_internal() {
    # Hypersim
    echo "Preparing Hypersim dataset (~147 GB)"
    nicr_sa_prepare_dataset \
        hypersim \
        ./datasets/hypersim \
        /datasets_nas/segmentation/hypersim/apple-hypersim \
        --additional-subsamples 2 5 10 20 \
        --n-processes 16 \
        --no-tilt-shift-conversion
}

prepare_datasets_internal() {
    rm -rf ./datasets
    mkdir -p /local/emsanet_datasets
    ln -s /local/emsanet_datasets ./datasets

    prepare_datasets_nyuv2_internal
    prepare_datasets_sunrgbd_internal
    prepare_datasets_hypersim_internal
}

prepare_datasets
# prepare_datasets_internal
