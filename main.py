import os
import time
import argparse
import copy
import json
import logging
import mlflow
import numpy as np

from tqdm import tqdm, trange
from datetime import datetime
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import init
from torch import optim
from torch.optim import Optimizer

from model import SASRec
from dataset import Dataset
from dataset import get_negative_samples
from utils import get_scheduler
from utils import *

StateDict = OrderedDict[str, torch.Tensor]


logger = logging.getLogger()


class Trainer:
    def __init__(
        self,
        dataset: Dataset,
        evaluate_k: int,
        model: SASRec,
        optimizer: Optimizer,
        max_lr: float,
        num_epochs: int,
        early_stop_epoch: int,
        warmup_ratio: float,
        use_scheduler: bool,
        scheduler_type: str,
        save_dir: str,
        resume_training: bool = False,
        device: str = "cpu",
    ) -> None:
        # Initialize device, evaluation metric, and directory for saving model checkpoints.
        self.device = device
        self.evaluate_k = evaluate_k
        self.save_dir = save_dir

        # Ensure the save directory exists; if not, create it.
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

        # Split the dataset into training, validation, and testing sets.
        self.train_data = dataset.user2items_train
        self.valid_data = dataset.user2items_valid
        self.test_data = dataset.user2items_test

        # Prepare data loaders for each dataset split.
        self.train_dataloader = dataset.get_dataloader(data=self.train_data)
        self.valid_dataloader = dataset.get_dataloader(data=self.valid_data,split="valid")
        self.test_dataloader = dataset.get_dataloader(data=self.test_data,split="test")
        # Map of user-item interactions used for negative sampling.
        self.positive2negatives = dataset.positive2negatives

        # Training configuration.
        self.max_lr = max_lr
        self.num_epochs = num_epochs
        self.early_stop_epoch = early_stop_epoch
        self.warmup_ratio = warmup_ratio
        self.scheduler_type = scheduler_type
        # Setup model and optimizer.
        self.model = model
        self.optimizer = optimizer
        # Configure the learning rate scheduler if enabled.
        self.use_scheduler = use_scheduler
        if self.use_scheduler:
            self.scheduler = get_scheduler(
                optimizer=optimizer,
                scheduler_type=scheduler_type,
                max_lr=max_lr,
                num_batches=len(self.train_dataloader),
                num_epochs=num_epochs,
                warmup_ratio=warmup_ratio,
            )
        else:
            self.scheduler = None

    def calculate_bce_loss(
        self,
        positive_idxs: torch.Tensor,
        negative_idxs: torch.Tensor,
        positive_logits: torch.Tensor,
        negative_logits: torch.Tensor,
    ) -> torch.Tensor:
        # Loss function for binary classification (BCE).
        loss_func = nn.BCEWithLogitsLoss()
        # Calculate loss for positive and negative samples.
        positive_logits = positive_logits[positive_idxs]
        positive_labels = torch.ones(size=positive_logits.shape).to(self.device)

        negative_logits = negative_logits[negative_idxs]
        negative_labels = torch.zeros(size=negative_logits.shape).to(self.device)

        positive_loss = loss_func(positive_logits, positive_labels)
        negative_loss = loss_func(negative_logits, negative_labels)

        return positive_loss + negative_loss

    def save_results(
        self,
        epoch: int,
        ndcg: float,
        model_state_dict: StateDict,
        optim_state_dict: StateDict,
        scheduler_state_dict: StateDict = None,
        save_name: str = "best",
    ) -> None:
        # Create a checkpoint dictionary with training artifacts.
        checkpoint = {
            f"{save_name}_epoch": epoch,
            f"{save_name}_ndcg": ndcg,
            f"{save_name}_model_state_dict": model_state_dict,
            f"{save_name}_optim_state_dict": optim_state_dict,
            f"{save_name}_scheduler_state_dict": scheduler_state_dict,
        }

        save_dir = os.path.join(self.save_dir, f"{save_name}_checkpoint.pt")
        torch.save(obj=checkpoint, f=save_dir)

    def train(self) -> (int, StateDict, StateDict):
        # Initialize tracking for the best model performance.
        best_ndcg = 0
        best_hit_rate = 0
        best_ndcg_epoch = 0
        best_hit_epoch = 0
        best_model_state_dict = None
        best_optim_state_dict = None
        best_scheduler_state_dict = None

        num_steps = 0
        epoch_pbar = trange(
            self.num_epochs,
            desc="Epochs: ",
            total=self.num_epochs,
        )
        # Iterate through each epoch and perform training.
        for epoch in epoch_pbar:
            self.model.train()  # Set the model to training mode.

            epoch_loss = 0
            # Iterate over training batches.
            train_pbar = tqdm(
                iterable=self.train_dataloader,
                desc="Training",
                total=len(self.train_dataloader),
            )
            for batch in train_pbar:
                self.model.zero_grad() # Zero out gradients to prevent accumulation from previous iterations.

                # Process batches to obtain sequences for training.
                positive_seqs = batch.clone()
                positive_idxs = torch.where(positive_seqs != 0) # Get indices of non-zero (positive) samples.

                batch[:, -1] = 0                    # Reset the last element to zero for all samples in the batch.
                input_seqs = batch.roll(shifts=1)   # Shift sequences to create model input.
                negative_seqs = get_negative_samples(
                    self.positive2negatives, positive_seqs
                )                                   # Sample negative sequences.
                negative_idxs = torch.where(negative_seqs != 0) # Get indices of non-zero (negative) samples.

                inputs = {
                    "input_seqs": input_seqs.to(self.device),
                    "positive_seqs": positive_seqs.to(self.device),
                    "negative_seqs": negative_seqs.to(self.device),
                }
                # Forward pass to compute logits for positive and negative samples.
                output = self.model(**inputs)

                positive_logits = output[0]
                negative_logits = output[1]
                # Compute loss and perform backpropagation.
                loss = self.calculate_bce_loss(
                    positive_idxs=positive_idxs,
                    negative_idxs=negative_idxs,
                    positive_logits=positive_logits,
                    negative_logits=negative_logits,
                )
                # Backward pass and optimizer step.
                loss.backward()
                epoch_loss += loss.item()   # Accumulate total loss for the epoch.
                self.optimizer.step()       # Update model weights.
                 # (Optional) Step the learning rate scheduler.
                if self.use_scheduler:
                    self.scheduler.step()

                num_steps += 1

            # Evaluate model performance after each epoch.
            ndcg, hit = self.evaluate()

            # Check if the current model is the best and save it.
            if ndcg >= best_ndcg:
                best_ndcg = ndcg
                best_ndcg_epoch = epoch
                best_model_state_dict = copy.deepcopy(x=self.model.state_dict())
                best_optim_state_dict = copy.deepcopy(x=self.optimizer.state_dict())
                # Save the best model checkpoint.
                if self.use_scheduler:
                    best_scheduler_state_dict = copy.deepcopy(
                        x=self.scheduler.state_dict()
                    )
                # Log training information and validate.
                logger.warning(f"New best. Saving to {self.save_dir}")
                self.save_results(
                    epoch=best_ndcg_epoch,
                    ndcg=best_ndcg,
                    model_state_dict=best_model_state_dict,
                    optim_state_dict=best_optim_state_dict,
                    scheduler_state_dict=best_scheduler_state_dict,
                )
                # (Optional) Log the best model to an external tracking system MLflow.
                mlflow.log_artifact(local_path=self.save_dir)

            if hit >= best_hit_rate:
                best_hit_rate = hit
                best_hit_epoch = epoch

            # Log epoch results and track metrics.
            epoch_result_msg = (
                f"\n\tEpoch {epoch}:"
                f"\n\t\tTraining Loss: {epoch_loss: 0.6f}, "
                f"\n\t\tnDCG@{self.evaluate_k}: {ndcg: 0.4f}, "
                f"\n\t\tHit@{self.evaluate_k}:  {hit: 0.4f}"
            )
            logger.info(epoch_result_msg)

            metrics = {
                "training-loss": epoch_loss,
                f"nDCG-{self.evaluate_k}": ndcg,
                f"Hit-{self.evaluate_k}": hit,
            }
            mlflow.log_metrics(
                metrics=metrics,
                step=epoch,
            )
            # Save the most recent model state.
            most_recent_model = self.model.state_dict()
            most_recent_optim = self.optimizer.state_dict()
            most_recent_scheduler = None

            if self.use_scheduler:
                most_recent_scheduler = self.scheduler.state_dict()

            self.save_results(
                epoch=epoch,
                ndcg=best_ndcg,
                model_state_dict=most_recent_model,
                optim_state_dict=most_recent_optim,
                scheduler_state_dict=most_recent_scheduler,
                save_name="most_recent",
            )
            mlflow.log_artifact(local_path=self.save_dir)

            # Track early stopping.
            if epoch - best_ndcg_epoch == self.early_stop_epoch:
                logger.warning(f"Stopping early at epoch {epoch}.")
                break
        # Log the best overall performance metrics.
        best_ndcg_msg = (
            f"Best nDCG@{self.evaluate_k} was {best_ndcg: 0.6f} "
            f"at epoch {best_ndcg_epoch}."
        )
        best_hit_msg = (
            f"Best Hit@{self.evaluate_k} was {best_hit_rate: 0.6f} "
            f"at epoch {best_hit_epoch}."
        )
        best_results_msg = "\n".join([best_ndcg_msg, best_hit_msg])
        logger.info(f"Best results:\n{best_results_msg}")

        return (best_ndcg_epoch, best_model_state_dict, best_optim_state_dict)

    def evaluate(
        self,
        mode: str = "valid",
        model: SASRec = None,
    ) -> (float, float):
        # Select the appropriate data loader based on the evaluation mode.
        if mode == "valid":
            dataloader = self.valid_dataloader
        else:
            dataloader = self.test_dataloader

        if model:
            self.model = model

        ndcg = 0
        hit = 0
        num_users = 0
        # Set model to evaluation mode.
        self.model.eval()
        # Disable gradient calculation for evaluation.
        with torch.no_grad():
            eval_pbar = tqdm(
                iterable=dataloader,
                desc=f"Evaluating for {mode}",
                total=len(dataloader),
            )
            # # Iterate over the evaluation dataset and compute evaluation metrics.
            for batch in eval_pbar:
                input_seqs, item_idxs = batch
                num_users += input_seqs.shape[0]

                inputs = {
                    "input_seqs": input_seqs.to(self.device),
                    "item_idxs": item_idxs.to(self.device),
                }
                # Forward pass to generate predictions.
                outputs = self.model(**inputs)

                logits = -outputs[0]

                ranks = logits.argsort().argsort()
                ranks = [r[0].item() for r in ranks]
                # Compute nDCG and Hit rate metrics based on the ranks.
                for rank in ranks:
                    if rank < self.evaluate_k:
                        ndcg += 1 / np.log2(rank + 2)
                        hit += 1

        ndcg /= num_users
        hit /= num_users

        return ndcg, hit

def main() -> None:
    # Parse command-line arguments for configurations.
    args = get_args()
    # Set a fixed seed for PyTorch to ensure reproducibility of results.
    torch.manual_seed(args.random_seed)
    mlflow.set_experiment(experiment_name=args.mlflow_experiment)

    # Get timestamp.
    time_right_now = time.time()
    timestamp = datetime.fromtimestamp(timestamp=time_right_now)
    timestamp = timestamp.strftime(format="%m-%d-%Y-%H%M")
    args.timestamp = timestamp

    # Get log file information and prepare output directories.
    data_name = args.data_filename.split(".txt")[0]
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    output_name = get_output_name(args, timestamp)
    args.mlflow_run_name = output_name
    log_filename = f"{output_name}.log"
    args.log_filepath = os.path.join(args.log_dir, log_filename)

    # Ensure the output directory exists; if not, create it.
    args.save_dir = os.path.join(args.output_dir, output_name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    # Save the runtime configuration to a JSON file for later reference.
    args_save_filename = os.path.join(args.save_dir, "args.json")
    with open(file=args_save_filename, mode="w") as f:
        json.dump(obj=vars(args), fp=f, indent=2)

    # If debugging mode is on, reduce the number of epochs to 1 
    # for quicker runs and set logging level to DEBUG.
    if args.debug:
        args.num_epochs = 1
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    # Logging basic configuration.
    log_msg_format = (
        "[%(asctime)s - %(levelname)s - %(filename)s: %(lineno)d] %(message)s"
    )
    handlers = [
        logging.FileHandler(filename=args.log_filepath),
        logging.StreamHandler(),
    ]
    logging.basicConfig(
        format=log_msg_format,
        level=log_level,
        handlers=handlers,
    )

    if args.debug:
        logger.warning("Debugging mode is turned on.")
    # Log initial settings and operations.
    logger.info(f"Starting main process with {data_name}...")
    logger.info(f"Output directory set to {args.save_dir}")
    logger.info(f"Logging file set to {args.log_filepath}")

    # Initialize dataset using settings provided in the arguments.
    dataset_args = DatasetArgs(args)
    dataset = Dataset(**vars(dataset_args))
    # Set number of items in the model's arguments based on the dataset.
    args.num_items = dataset.num_items
    # Initialize the recommendation model.
    model_args = ModelArgs(args)
    model = SASRec(**vars(model_args))
    # Initialize model parameters using Xavier uniform distribution.
    for param in model.parameters():
        try:
            init.xavier_uniform_(param.data)
        except ValueError:
            continue

    model = model.to(args.device)
    # Configure the optimizer with settings from the arguments.
    optimizer_args = OptimizerArgs(args)
    optimizer = optim.Adam(params=model.parameters(), **vars(optimizer_args))
    # Initialize the trainer with the dataset, model, optimizer, and additional arguments
    trainer_args = TrainerArgs(args)
    trainer = Trainer(
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        **vars(trainer_args),
    )
    # Log the parameters for the MLflow tracking.
    mlflow.log_params(vars(args))

    mlflow.end_run()

    with mlflow.start_run(run_name=args.mlflow_run_name):
        mlflow.log_artifact(local_path=args.log_filepath, artifact_path="logs")
        best_results = trainer.train()
        # Start the training process and capture the best results.
        best_ndcg_epoch, best_model_state_dict, _ = best_results

        # Evaluate the model using the best state dict on the test set.
        model.load_state_dict(best_model_state_dict)
        logger.info(f"Testing with model checkpoint from epoch {best_ndcg_epoch}...")
        test_ndcg, test_hit_rate = trainer.evaluate(mode="test", model=model)
        # Format and log the final test performance.
        test_ndcg_msg = f"Test nDCG@{trainer_args.evaluate_k} is {test_ndcg: 0.6f}."
        test_hit_msg = f"Test Hit@{trainer_args.evaluate_k} is {test_hit_rate: 0.6f}."
        test_result_msg = "\n".join([test_ndcg_msg, test_hit_msg])
        logger.info(f"\n{test_result_msg}")


if __name__ == "__main__":
    main()
