import json
import logging
import numpy as np
import os
import wandb
from tqdm import tqdm
import math
import pandas as pd

import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler, RobertaModel, AutoConfig

from fair import FairConfig, FairModel, FairForRetrieval
from utils import (
    load_tokenizer,
    model_summary,
    EarlyStopController,
    MixedPrecisionManager,
    save_eval_details
)
from .data import prepare_dataset
from metrics import compute_map


def run_retrieval(args):
    logging.info(f"Task: {args.task}")
    # --------------------------------------------------
    # Prepare tokenizer and model
    # --------------------------------------------------
    logging.info(f"***** Loading tokenizer and model *****")
    # tokenizer
    tokenizer, cfg_vocab, dfg_vocab = load_tokenizer(args)

    # model
    if args.model_path is None:
        if args.model_size == "small":
            from fair.configuration_fair_small import FairConfig
        else:
            from fair.configuration_fair import FairConfig
        config = FairConfig()
        model = FairForRetrieval(config)
    else:
        assert os.path.exists(args.model_path)
        if os.path.isdir(args.model_path):
            if args.model_size == "small":
                from fair.configuration_fair_small import FairConfig
            else:
                from fair.configuration_fair import FairConfig
            config = FairConfig()
            model = FairForRetrieval(config)
            fair = FairModel.from_pretrained(args.model_path)
            model.fair = fair
        else:
            model = torch.load(args.model_path)

    model_summary(model)
    model.to(args.device)
    if args.num_devices > 1:
        model = torch.nn.DataParallel(model)

    # --------------------------------------------------
    # Train and valid
    # --------------------------------------------------
    if not args.only_test:
        model = train(
            args, model=model, tokenizer=tokenizer, cfg_vocab=cfg_vocab, dfg_vocab=dfg_vocab
        )
        torch.cuda.empty_cache()

    # --------------------------------------------------
    # predict
    # --------------------------------------------------
    test_results = test(
        args, model=model, tokenizer=tokenizer, cfg_vocab=cfg_vocab, dfg_vocab=dfg_vocab
    )


def train(args, model, tokenizer, cfg_vocab, dfg_vocab):
    logging.info("***** Loading Datasets *****")
    train_dataset, train_examples = prepare_dataset(
        args=args, tokenizer=tokenizer, cfg_vocab=cfg_vocab, dfg_vocab=dfg_vocab, split="train"
    )
    logging.info(f"Train dataset is prepared, size: {len(train_dataset)}")
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    valid_dataset, valid_examples = prepare_dataset(
        args=args, tokenizer=tokenizer, cfg_vocab=cfg_vocab, dfg_vocab=dfg_vocab, split="valid"
    )
    logging.info(f"Valid dataset is prepared, size: {len(valid_dataset)}")
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    logging.info("***** Preparing Training Utils *****")
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate
    )

    # calculate max steps
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_epochs * num_update_steps_per_epoch

    if args.warmup_steps >= 1:
        args.warmup_steps = int(args.warmup_steps)
    elif args.warmup_steps >= 0:
        args.warmup_steps = int(args.warmup_steps * args.max_train_steps)
    else:
        raise ValueError(f"Invalid warmup steps: {args.warmup_steps}")
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # mixed precision
    amp = MixedPrecisionManager(activated=args.fp16)

    # early stop
    early_stop = EarlyStopController(
        patience=args.patience,
        best_model_dir=os.path.join(args.model_dir, f"best_{args.metric_for_best_model}"),
        higher_is_better=True,
    )

    # batch size per device, total batch size
    if args.num_devices > 1:
        batch_size_per_device = args.batch_size // args.num_devices
        if batch_size_per_device * args.num_devices != args.batch_size:
            raise ValueError(
                f"The total batch size {args.batch_size=} is not an integer multiple "
                f"of the device count: {args.num_devices}"
            )
    else:
        batch_size_per_device = args.batch_size
    total_batch_size = args.batch_size * args.gradient_accumulation_steps

    logging.info("***** Training *****")
    logging.info(f"  Num examples = {len(train_dataset)}")
    logging.info(f"  Num Epochs = {args.num_epochs}")
    logging.info(f"  Batch size per device = {batch_size_per_device}")
    logging.info(
        f"  Total train batch size (w. parallel & accumulation) = {total_batch_size}"
    )
    logging.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
    )
    logging.info(f"  Total optimization steps = {args.max_train_steps}")

    completed_steps = 0
    model.zero_grad()

    train_bar = tqdm(
        range(args.max_train_steps),
        total=args.max_train_steps,
        ascii=True,
    )
    for epoch in range(args.num_epochs):
        model.train()

        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            with amp.context():

                for k, v in batch.items():
                    batch[k] = v.to(args.device)

                loss, vector = model(**batch)

                if args.num_devices > 1:
                    loss = loss.mean()

                loss = loss / args.gradient_accumulation_steps

            amp.backward(loss)

            if (
                    (step + 1) % args.gradient_accumulation_steps == 0
                    or step == len(train_dataloader) - 1
            ):
                amp.step(
                    model=model,
                    optimizer=optimizer,
                    max_grad_norm=args.max_grad_norm,
                )
                lr_scheduler.step()
                completed_steps += 1

                train_bar.update(1)

            if (step + 1) % (args.gradient_accumulation_steps * args.logging_steps) == 0:
                logging.info(
                    {
                        "global_step": completed_steps,
                        "epoch": completed_steps / num_update_steps_per_epoch,
                        "loss": loss.item(),
                    }
                )
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/epoch": completed_steps / num_update_steps_per_epoch,
                        "train/learning_rate": lr_scheduler.optimizer.param_groups[0]["lr"]
                    },
                    step=completed_steps,
                )

            if completed_steps >= args.max_train_steps:
                break

        logging.info("Start validation")
        valid_results = __eval(
            args=args,
            model=model,
            dataloader=valid_dataloader,
            split="valid",
            epoch=epoch,
        )

        early_stop(
            score=valid_results[f"valid/{args.metric_for_best_model}"],
            model=model,
            epoch=epoch,
            step=completed_steps,
        )
        if not early_stop.hit:
            logging.info(
                f"Early stopping counter: {early_stop.counter}/{early_stop.patience}"
            )

        if early_stop.early_stop:
            logging.info(f"Early stopping is triggered")
            break

    logging.info("End of training")

    # load best model at end of training
    model = early_stop.load_best_model()
    model.to(args.device)
    if args.num_devices > 1:
        model = torch.nn.DataParallel(model)

    return model


def test(args, model, tokenizer, cfg_vocab, dfg_vocab):
    logging.info("***** Testing *****")

    test_dataset, test_examples = prepare_dataset(
        args=args, tokenizer=tokenizer, cfg_vocab=cfg_vocab, dfg_vocab=dfg_vocab, split="test"
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    logging.info(f"Test dataset is prepared, size: {len(test_dataset)}")

    test_results = __eval(
        args=args, model=model, dataloader=test_dataloader, split="test"
    )
    logging.info("End of testing")

    return test_results


def __eval(args, model, dataloader, split, epoch=None):
    assert split in ["valid", "test"]
    assert split == "test" or epoch is not None

    # statistics
    num_examples = 0
    num_steps = 0
    all_loss = []

    # used for computing map
    all_vectors = []
    all_labels = []

    eval_bar = tqdm(dataloader, total=len(dataloader), ascii=True)
    model.eval()
    for step, batch in enumerate(eval_bar):

        with torch.no_grad():
            for k, v in batch.items():
                batch[k] = v.to(args.device)

            labels = batch.get("labels")

            loss, vector = model(**batch)
            all_loss.append(loss.mean().item())
            all_vectors.append(vector.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        num_examples += len(labels)
        num_steps += 1

    all_vectors = np.concatenate(all_vectors, 0)
    all_labels = np.concatenate(all_labels, 0)
    eval_loss = np.mean(all_loss)

    results = compute_map(vectors=all_vectors, labels=all_labels, prefix=split)
    results.update({f"{split}/loss": eval_loss})
    wandb.log(results)

    results.update({
        f"{split}/num_examples": num_examples,
        f"{split}/num_steps": num_steps,
    })
    logging.info(results)

    # make sure that the metric for selecting best model is in the results when validating
    if split == "valid" and f"{split}/{args.metric_for_best_model}" not in results:
        raise ValueError(f"The metric for selecting best model is set to {args.metric_for_best_model}, "
                         f"which is, however, not found in to validation results.")

    logging.info(f"Start saving {split} results...")
    # save results to json file
    save_dir = os.path.join(
        args.eval_dir, f"valid_epoch_{epoch}" if split == "valid" else "test"
    )
    os.makedirs(save_dir)
    with open(os.path.join(save_dir, "results.json"), mode="w", encoding="utf-8") as f:
        json.dump(results, f)

    if split == "test" or args.save_valid_details:
        # save summary
        scores = np.matmul(all_vectors, all_vectors.T)
        np.fill_diagonal(scores, -1000000)
        sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]

        details = {
            "Index": [],
            "Label": [],
            "Rank": [],
            "Retrieved_index": [],
            "Retrieved_label": [],
            "Success": []
        }
        for i in tqdm(range(num_examples), desc="Gathering", ascii=True):
            label = all_labels[i]
            # we only save up to top-20 results due to the extreme large number of summary
            for j in range(min(num_examples - 1, 20)):
                retrieved_index = sort_ids[i][j]
                retrieved_label = all_labels[retrieved_index]

                details["Index"].append(i)
                details["Label"].append(label)
                details["Rank"].append(j)
                details["Retrieved_index"].append(retrieved_index)
                details["Retrieved_label"].append(retrieved_label)
                details["Success"].append(retrieved_label == label)

        df = pd.DataFrame.from_dict(details)

        # save testing results
        save_eval_details(save_dir=save_dir, detail_df=df)

    logging.info(f"{split.capitalize()} results are saved to {save_dir}")

    model.train()

    return results
