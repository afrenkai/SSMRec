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


def get_output_name(args, timestamp):
    base_name = os.path.splitext(args.data_filename)[0]
    return f"sasrec-{base_name}_lr-{args.lr}_bs-{args.batch_size}_{timestamp}"


def get_args():
    parser = argparse.ArgumentParser(
        description="Configure the parameters for the SASRec model."
    )
    parser.add_argument(
        "--model", default="ssm", help="use SSM or SA for the base model"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode for verbose output.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Enable saving of the model checkpoints.",
    )

    parser.add_argument(
        "--data_filename", default="amazon_beauty.txt", help="Name of data file."
    )
    default_log_dir = os.getenv("LOG_DIR", "./logs")
    parser.add_argument(
        "--log_dir", default=default_log_dir, help="Directory to save logs"
    )
    parser.add_argument(
        "--data_root",
        default=os.getenv("DATA_ROOT", "./data"),
        help="Root directory for datasets.",
    )
    parser.add_argument(
        "--output_dir", default="./outputs", help="Directory to save outputs."
    )
    parser.add_argument("--resume_dir", default="")

    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=50,
        help="Maximum number of items to see. Denoted by $n$ in the paper.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training."
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=50, help="Dimensionality of embedding matrix."
    )
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=2,
        help="Number of self-attention -> FFNN blocks to stack.",
    )
    parser.add_argument(
        "--dropout_p",
        type=float,
        default=0.5,
        help="Dropout rate applied to embedding layer and FFNN.",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=1,
        help="Number of attention heads in self-attention.",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=50, help="Number of epochs to train."
    )
    parser.add_argument(
        "--share_item_emb",
        action="store_true",
        help="Whether or not to use item matrix for prediction layer.",
    )

    parser.add_argument("--device", default="")
    parser.add_argument(
        "--beta1", default=0.9, type=float, help="Beta1 argument for Adam optimizer."
    )
    parser.add_argument(
        "--beta2", default=0.99, type=float, help="Beta2 argument for Adam optimizer."
    )
    parser.add_argument(
        "--eps", default=1e-8, type=float, help="Epsilon value for Adam optimizer."
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay rate for Adam optimizer.",
    )
    parser.add_argument(
        "--use_scheduler", action="store_true", help="Use a learning rate scheduler."
    )
    parser.add_argument(
        "--scheduler_type", default="onecycle", help="Type of scheduler to use."
    )
    parser.add_argument(
        "--evaluate_k",
        type=int,
        default=10,
        help="Top-K for evaluation metrics (nDCG@K, HR@K).",
    )
    parser.add_argument(
        "--early_stop_epoch",
        type=int,
        default=20,
        help="Epoch to stop training if no improvement.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="Ratio of total steps for warmup.",
    )
    parser.add_argument("--resume_training", action="store_true", default=False)

    parser.add_argument("--mlflow_experiment", default="sasrec-pytorch-experiments")
    parser.add_argument("--mlflow_run_name", default="")

    args = parser.parse_args()
    args.device = get_device()
    return args


class DatasetArgs:
    def __init__(self, args: argparse.Namespace) -> None:
        args.data_root = os.path.expanduser(path=args.data_root)
        if not os.getcwd().endswith("src") and os.getcwd().endswith("sasrec-pytorch"):
            args.data_root = "./data"
        self.data_filepath = os.path.join(args.data_root, args.data_filename)
        assert os.path.exists(
            self.data_filepath
        ), f"{self.data_filepath} does not exist!"
        self.batch_size = args.batch_size
        self.max_seq_len = args.max_seq_len
        self.debug = args.debug


class ModelArgs:
    def __init__(self, args: argparse.Namespace) -> None:
        self.device = args.device
        self.num_items = args.num_items
        self.num_blocks = args.num_blocks
        self.max_seq_len = args.max_seq_len
        self.hidden_dim = args.hidden_dim
        self.dropout_p = args.dropout_p
        self.num_heads = args.num_heads
        self.share_item_emb = args.share_item_emb


class OptimizerArgs:
    def __init__(self, args: argparse.Namespace) -> None:
        self.lr = args.lr
        self.betas = (args.beta1, args.beta2)
        self.eps = args.eps
        self.weight_decay = args.weight_decay


class TrainerArgs:
    def __init__(self, args: argparse.Namespace) -> None:
        self.device = args.device
        self.evaluate_k = args.evaluate_k
        self.max_lr = args.lr
        self.num_epochs = args.num_epochs
        self.early_stop_epoch = args.early_stop_epoch
        self.use_scheduler = args.use_scheduler
        self.warmup_ratio = args.warmup_ratio
        self.scheduler_type = args.scheduler_type
        self.resume_training = args.resume_training
        self.save_dir = args.save_dir
