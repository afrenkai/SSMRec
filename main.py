import os
import time
import json
import logging
import mlflow
import torch
from torch import optim
from torch.nn import init
from datetime import datetime

from model import SASRec
from model_ssm import SSMRec
from dataset import Dataset
from trainer import Trainer
from config import (
    get_device,
    get_args,
    get_output_name,
    TrainerArgs,
    OptimizerArgs,
    DatasetArgs,
    ModelArgs,
)
#TODO: detect ifd ampere plus and disable if not
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger()


def main() -> None:
    args = get_args()
    device = get_device()
    torch.manual_seed(args.random_seed)
    mlflow.set_experiment(experiment_name=args.mlflow_experiment)

    time_right_now = time.time()
    timestamp = datetime.fromtimestamp(timestamp=time_right_now)
    timestamp = timestamp.strftime(format="%m-%d-%Y-%H%M")
    args.timestamp = timestamp

    data_name = args.data_filename.split(".txt")[0]
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    output_name = get_output_name(args, timestamp)
    args.mlflow_run_name = output_name
    log_filename = f"{output_name}.log"
    args.log_filepath = os.path.join(args.log_dir, log_filename)

    args.save_dir = os.path.join(args.output_dir, output_name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    args_save_filename = os.path.join(args.save_dir, "args.json")
    with open(file=args_save_filename, mode="w") as f:
        json.dump(obj=vars(args), fp=f, indent=2)

    if args.debug:
        args.num_epochs = 1
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

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
    logger.info(f"Starting main process with {data_name}...")
    logger.info(f"Using device: {device}")
    logger.info(f"model name: {args.model}")
    logger.info(f"Output directory set to {args.save_dir}")
    logger.info(f"Logging file set to {args.log_filepath}")

    dataset_args = DatasetArgs(args)
    dataset = Dataset(**vars(dataset_args))
    args.num_items = dataset.num_items
    
    model_args = ModelArgs(args)
    if args.model == "ssm":
        model = SSMRec(**vars(model_args))
    else:
        model = SASRec(**vars(model_args))
        
    for param in model.parameters():
        try:
            init.xavier_uniform_(param.data)
        except ValueError:
            continue

    model = model.to(device)
    

    try:
        model = torch.compile(model, mode="reduce-overhead")
        logger.info("Model compiled with torch.compile for faster training")
    except Exception as e:
        logger.info(f"torch.compile not available or failed: {e}")
    
    print(device)
    optimizer_args = OptimizerArgs(args)
    optimizer = optim.Adam(params=model.parameters(), **vars(optimizer_args))
    
    trainer_args = TrainerArgs(args)
    trainer = Trainer(
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        **vars(trainer_args),
    )
    mlflow.log_params(vars(args))

    mlflow.end_run()

    with mlflow.start_run(run_name=args.mlflow_run_name):
        mlflow.log_artifact(local_path=args.log_filepath, artifact_path="logs")
        best_results = trainer.train()
        best_ndcg_epoch, best_model_state_dict, _ = best_results

        model.load_state_dict(best_model_state_dict)
        logger.info(f"Testing with model checkpoint from epoch {best_ndcg_epoch}...")
        test_ndcg, test_hit_rate = trainer.evaluate(mode="test", model=model)
        test_ndcg_msg = f"Test nDCG@{trainer_args.evaluate_k} is {test_ndcg: 0.6f}."
        test_hit_msg = f"Test Hit@{trainer_args.evaluate_k} is {test_hit_rate: 0.6f}."
        test_result_msg = "\n".join([test_ndcg_msg, test_hit_msg])
        logger.info(f"\n{test_result_msg}")


if __name__ == "__main__":
    main()
