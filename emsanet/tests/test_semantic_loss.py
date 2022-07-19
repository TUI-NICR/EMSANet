# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
"""
import numpy as np
import torch
from torch import nn

from nicr_mt_scene_analysis.loss.ce import CrossEntropyLossSemantic

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# copied from: https://github.com/TUI-NICR/ESANet/blob/main/src/utils.py#L18-L50
class CrossEntropyLossPrevious(nn.Module):
    def __init__(self, device, weight):
        super(CrossEntropyLossPrevious, self).__init__()
        self.weight = torch.tensor(weight).to(device)
        self.num_classes = len(self.weight) + 1  # +1 for void
        if self.num_classes < 2 ** 8:
            self.dtype = torch.uint8
        else:
            self.dtype = torch.int16
        self.ce_loss = nn.CrossEntropyLoss(
            torch.from_numpy(np.array(weight)).float(),
            reduction='none',
            ignore_index=-1
        )
        self.ce_loss.to(device)

    def forward(self, inputs_scales, targets_scales):
        losses = []
        for inputs, targets in zip(inputs_scales, targets_scales):
            # mask = targets > 0
            targets_m = targets.clone()
            targets_m -= 1
            loss_all = self.ce_loss(inputs, targets_m.long())

            number_of_pixels_per_class = \
                torch.bincount(targets.flatten().type(self.dtype),
                               minlength=self.num_classes)
            divisor_weighted_pixel_sum = \
                torch.sum(number_of_pixels_per_class[
                          1:] * self.weight)  # without void
            losses.append(torch.sum(loss_all) / divisor_weighted_pixel_sum)
            # losses.append(torch.sum(loss_all) / torch.sum(mask.float()))

        return losses


def test_loss():
    """
    make sure that new loss implementation outputs the same values as the old
    loss implementation
    """
    class_weights = torch.tensor(
        [0.2650426, 0.5533999, 0.42025763, 0.34482047, 0.7993162,
         0.49264285, 1.1026958, 0.78996897, 0.76780474, 0.36996013,
         1.6053797, 0.97266424, 0.63303965, 0.73651886, 0.92407864,
         0.59753835, 0.4705898, 1.7916499, 0.61840767, 1.1446692,
         1.1642636, 1.081512, 1.8748288, 0.6763455, 1.0289167,
         4.0649543, 1.5289997, 0.42058772, 3.60466, 0.53412074,
         1.246997, 2.2661245, 0.9652696, 3.0297952, 5.316681,
         1.0555762, 6.7779245, 1.0640355, 1.2999853, 1.1953188],
    )

    # loss object from new loss implementation
    loss_object = CrossEntropyLossSemantic(
        weights=class_weights.to(device),
        label_smoothing=0.0,
        weighted_reduction=True
    )

    # loss object from old loss implementation
    loss_object_previous = CrossEntropyLossPrevious(
        device=device,
        weight=class_weights
    )

    # generate random prediction and target
    pred = tuple(
        torch.rand(size=(2, 40, int(480 / (2 ** i)), int(640 / (2 ** i))),
                   device=device)
        for i in [0, 3, 4, 5]
    )

    target = tuple(
        torch.randint(size=(2, int(480 / (2 ** i)), int(640 / (2 ** i))),
                      low=0,
                      high=40,
                      device=device)
        for i in [0, 3, 4, 5]
    )

    # compute loss with new as well as with old loss object
    loss = loss_object(pred, target)
    loss_previous = loss_object_previous(pred, target)

    # test if the computed losses are the same between both implementations
    assert torch.allclose(
        torch.tensor([loss[i][0] for i in range(4)]),
        torch.tensor([loss_previous[i] for i in range(4)])
    )
