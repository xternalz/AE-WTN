# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from functools import partial
import torch

from .lr_scheduler import WarmupMultiStepLR, CosineAnnealingLR

def make_optimizer(cfg, model):
    params = [[],[]]
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if "wtn" not in key:
            params[0] += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        else:
            if "wtn_enc.1." in key or "wtn_dec.1." in key:
                params[1] += [{"params": [value], "weight_decay": 0}]
            else:
                params[1] += [{"params": [value]}]

    optimizers = [torch.optim.SGD(params[0], lr, momentum=cfg.SOLVER.MOMENTUM)]
    optimizers.append(torch.optim.AdamW(params[1], weight_decay=0.1))
    return optimizers


def make_lr_scheduler(cfg, optimizers):
    lrs0 = WarmupMultiStepLR(
        optimizers[0],
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
    )
    lrs1 = WarmupMultiStepLR(
        optimizers[1],
        cfg.SOLVER.STEPS,
        (1/10.0)**0.5,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
    )
    return [lrs0, lrs1]
