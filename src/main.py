import argparse
import os
import time
import wandb
import random
import numpy as np
import logging
import sys
from prettytable import PrettyTable

from tasks import run_classification, run_retrieval, run_k_fold
from args import add_args, check_args


def main():
    # global system settings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["NCCL_P2P_DISABLE"] = "1"

    # parse arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register("type", "bool", lambda v: v.lower() in ["yes", "true", "t", "1", "y"])

    add_args(parser)
    args = parser.parse_args()

    # check args
    check_args(args)

    # prepare some preliminary arguments
    if args.run_name is None:
        args.run_name = args.task
    args.unique_run_name = "{}_{}".format(
        args.run_name,
        time.strftime("%Y%m%d_%H%M%S", time.localtime())
    )

    # outputs and savings
    args.output_dir = os.path.join("..", "outputs", args.unique_run_name)  # root of outputs/savings
    args.model_dir = os.path.join(args.output_dir, "model")
    # args.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")  # dir of saving models
    # args.best_checkpoint_dir = os.path.join(args.checkpoint_dir, "best")  # best checkpoint
    args.eval_dir = os.path.join(args.output_dir, "evaluations")  # dir of saving evaluation results
    # args.tb_dir = os.path.join(args.output_dir, "tb_runs")  # dir of tracking running with tensorboard
    for d in [args.output_dir, args.model_dir, args.eval_dir]:
        os.makedirs(d, exist_ok=True)

    # logging, log to both console and file, log debug-level to file
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    # terminal printing
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level=logging.INFO)
    console.setFormatter(
        logging.Formatter(
            "[%(asctime)s] - %(levelname)s: %(message)s"
        )
    )
    logger.addHandler(console)
    # logging file
    file = logging.FileHandler(os.path.join(args.output_dir, "logging.log"))
    file.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s | %(filename)s | line %(lineno)d] - %(levelname)s: %(message)s"
    )
    file.setFormatter(formatter)
    logger.addHandler(file)

    logger.info("***** Initializing Environments *****")

    # device and parallelism
    # this statement must be before the `import torch`
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    import torch

    # this is to solve the "too many open files" error
    # when num_workers of dataloader greater than 0
    # https://github.com/pytorch/pytorch/issues/11201#issuecomment-421146936
    torch.multiprocessing.set_sharing_strategy("file_system")

    args.use_cuda = torch.cuda.is_available() and not args.no_cuda
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    logging.info(f"Use CUDA: {args.use_cuda}")
    if args.use_cuda:
        if args.cuda_visible_devices is not None:
            logging.info(f"Visible CUDA device ids: {args.cuda_visible_devices}")
        args.num_devices = torch.cuda.device_count()
        logging.info(f"Number of available gpus: {args.num_devices}")
    else:
        args.num_devices = 1

    args.device = torch.device("cuda" if args.use_cuda else "cpu")
    logging.info(f"Device: {args.device}")

    args.fp16 = args.fp16 and args.use_cuda
    logging.info(f"Use fp16 mixed precision: {args.fp16}")

    # setup random seed
    if args.random_seed > 0:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        logging.info(f"Random seed: {args.random_seed}")

    # log command and configs
    logger.debug("COMMAND: {}".format(" ".join(sys.argv)))

    config_table = PrettyTable()
    config_table.field_names = ["Configuration", "Value"]
    config_table.align["Configuration"] = "l"
    config_table.align["Value"] = "l"
    for config, value in vars(args).items():
        config_table.add_row([config, str(value)])
    logger.debug("Configurations:\n{}".format(config_table))

    # wandb
    if args.wandb_mode == "online":
        assert os.path.exists("../wandb_api.key")
        with open("../wandb_api.key", mode="r", encoding="utf-8") as f:
            os.environ["WANDB_API_KEY"] = f.read().strip()
    wandb_run = wandb.init(
        project="FAIR",
        dir=args.output_dir,
        name=args.run_name,
        mode=args.wandb_mode,
        config=vars(args)
    )

    # run task by type
    if args.task == "pre-train":
        raise NotImplementedError
    elif args.task in ["cls-poj104"]:
        run_classification(args)
    elif args.task in ["device-nvidia", "device-amd",
                       "thread-fermi", "thread-kepler", "thread-cypress", "thread-tahiti"]:
        run_k_fold(args)
    elif args.task in ["retrieval-poj104"]:
        run_retrieval(args)

    wandb_run.finish()


if __name__ == "__main__":
    main()
