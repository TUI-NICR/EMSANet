# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import pytest
import time

from nicr_scene_analysis_datasets.utils.testing import DATASET_PATH_DICT

from emsanet.args import ArgParserEMSANet
from emsanet.data import get_datahelper
from emsanet.preprocessing import get_preprocessor
from emsanet.data import KNOWN_DATASETS


@pytest.mark.parametrize('dataset', KNOWN_DATASETS)
def test_data_helper(dataset):
    """Test data helper"""
    # get args
    parser = ArgParserEMSANet()
    if 'coco' == dataset:
        input_modalities = ('rgb',)
    else:
        input_modalities = ('rgb', 'depth')
    args = parser.parse_args(
        ['--dataset', dataset,
         '--dataset-path', DATASET_PATH_DICT[dataset],
         '--input-modalities', *input_modalities],
        verbose=False)

    data = get_datahelper(args)
    for idx, batch in enumerate(data.train_dataloader):
        assert batch is not None
        if idx == 10:
            break

    for idx, batch in enumerate(data.valid_dataloaders[0]):
        assert batch is not None
        if idx == 10:
            break


def test_data_caching():
    """Test dataset caching"""
    dataset = "nyuv2"
    dataset_path = DATASET_PATH_DICT[dataset]
    n_worksers = 4

    parser = ArgParserEMSANet()
    args = parser.parse_args('', verbose=False)

    # replace some args
    args.dataset = dataset
    args.dataset_path = dataset_path
    args.n_workers = n_worksers
    args.cache_dataset = True

    data = get_datahelper(args)
    data.set_valid_preprocessor(
        get_preprocessor(
            args,
            dataset=data.datasets_valid[0],
            phase='test',
            multiscale_downscales=None
        )
    )

    # iteration should be faster in later runs (after cache of all workers is
    # ready)
    simple_sums = []
    durations = []
    data_loader = data.valid_dataloaders[0]
    for _ in range(4*n_worksers):
        start = time.time()
        sum = 0
        for sample in data_loader:
            for rgb in sample['rgb']:
                sum += rgb.numpy().sum()
        end = time.time()

        simple_sums.append(sum)
        durations.append(end-start)

        # print(simple_sums[-1], durations[-1])

    # note that all workers have to cache the dataset
    assert all(d < durations[0] for d in durations[-2*n_worksers:])
    assert all(s == simple_sums[0] for s in simple_sums[-2*n_worksers:])
