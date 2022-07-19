# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Union

from torch.optim import Adam
from torch.optim import SGD


KNOWN_OPTIMIZERS = ('sgd', 'adam')


OptimizerType = Union[Adam, SGD]


def get_optimizer(args, parameters) -> OptimizerType:
    name = args.optimizer
    lr = args.learning_rate
    weight_decay = args.weight_decay
    momentum = args.momentum

    name = name.lower()
    if name not in KNOWN_OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: '{name}'")

    if 'sgd' == args.optimizer:
        optimizer = SGD(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=True
        )
    elif 'adam' == args.optimizer:
        optimizer = Adam(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )

    return optimizer
