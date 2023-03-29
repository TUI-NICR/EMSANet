# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import os
import shutil

import pytest


def pytest_addoption(parser):
    parser.addoption('--keep-files', action='store_true', default=False)
    parser.addoption('--force-onnx-export', action='store_true', default=False)
    parser.addoption('--show-results', action='store_true', default=False)


def pytest_configure(config):
    # see: nicr_mt_scene_analysis/testing/__init__.py
    if config.getoption('--force-onnx-export'):
        os.environ['EXPORT_ONNX_MODELS'] = str(True)
    if config.getoption('--show-results'):
        os.environ['SHOW_RESULTS'] = str(True)


@pytest.fixture(scope='session')
def keep_files(request):
    return request.config.getoption('--keep-files')


@pytest.fixture(scope='session')
def tmp_path(tmpdir_factory, keep_files):
    # see: https://docs.pytest.org/en/6.2.x/reference.html#tmpdir-factory
    # use '--basetemp' to change default path
    # -> BE AWARE <- --basetemp is cleared on start

    path = tmpdir_factory.mktemp('emsanet')
    print(f"\nWriting temporary files to '{path}'")
    if keep_files:
        print("Files are kept and require to be deleted manually.'")

    yield path

    # teardown (delete if it was created)
    if os.path.exists(path) and not keep_files:
        shutil.rmtree(path)
