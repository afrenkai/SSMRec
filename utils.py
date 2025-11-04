import argparse
import os
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_scheduler(
    optimizer: Optimizer,
    scheduler_type: str,
    max_lr: float,
    num_batches: int,
    num_epochs: int,
    warmup_ratio: float,
):
    if scheduler_type == "onecycle":
        total_steps = int(num_batches * num_epochs)
        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=warmup_ratio,
            anneal_strategy="linear",
        )
    else:
        raise NotImplementedError(
            f"Scheduler type {scheduler_type} is not implemented."
        )
