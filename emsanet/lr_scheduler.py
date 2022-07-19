# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from torch.optim.lr_scheduler import OneCycleLR


KNOWN_LR_SCHEDULERS = ('onecycle', )


LrSchedulerType = OneCycleLR


def get_lr_scheduler(args, optimizer) -> LrSchedulerType:
    name = args.learning_rate_scheduler
    n_epochs = args.n_epochs

    name = name.lower()
    if name not in KNOWN_LR_SCHEDULERS:
        raise ValueError(f"Unknown learning rate scheduler: '{name}'")

    if 'onecycle' == name:
        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=[i['lr'] for i in optimizer.param_groups],
            total_steps=n_epochs,
            div_factor=25,
            pct_start=0.1,
            anneal_strategy='cos',
            final_div_factor=1e4
        )

    return lr_scheduler
