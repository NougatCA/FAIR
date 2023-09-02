import logging
import numpy as np
import os
import pandas as pd
import wandb
from sklearn.model_selection import KFold
from tqdm import tqdm
import json
import math

import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler
from torch.utils.data import Subset

from fair import FairConfig, FairForSequenceClassification, FairModel
from utils import (
    load_tokenizer,
    model_summary,
    save_eval_details,
    MixedPrecisionManager,
    EarlyStopController,
    LabelSmoother,
)
from .data import prepare_dataset
from metrics import (
    compute_acc,
    compute_p_r_f1,
    speed_up_for_device_mapping,
    speed_up_for_thread_coarsening
)


def run_k_fold(args):
    logging.info(f"Task: {args.task}")
    # --------------------------------------------------
    # Prepare tokenizer and model
    # --------------------------------------------------
    logging.info(f"***** Loading tokenizer and model *****")
    # tokenizer
    tokenizer, cfg_vocab, dfg_vocab = load_tokenizer(args)

    # model
    def __init_model():
        if args.model_path is None:
            if args.model_size == "small":
                from fair.configuration_fair_small import FairConfig
            else:
                from fair.configuration_fair import FairConfig
            config = FairConfig()
            config.num_labels = args.num_labels
            __model = FairForSequenceClassification(config)
        else:
            assert os.path.exists(args.model_path)
            if os.path.isdir(args.model_path):
                if args.model_size == "small":
                    from fair.configuration_fair_small import FairConfig
                else:
                    from fair.configuration_fair import FairConfig
                config = FairConfig()
                config.num_labels = args.num_labels
                __model = FairForSequenceClassification(config)
                fair = FairModel.from_pretrained(args.model_path)
                __model.fair = fair
            else:
                __model = torch.load(args.model_path)
        model_summary(__model)
        __model.to(args.device)
        if args.num_devices > 1:
            __model = torch.nn.DataParallel(__model)
        return __model

    # data
    logging.info("***** Loading Datasets *****")
    dataset, examples = prepare_dataset(
        args=args, tokenizer=tokenizer, cfg_vocab=cfg_vocab, dfg_vocab=dfg_vocab
    )
    logging.info(f"Dataset is prepared, size: {len(dataset)}")

    # start k-fold
    spliter = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.random_seed)
    indices = spliter.split(dataset)
    all_acc = []
    all_speed_up = []
    for fold, (train_indices, valid_indices) in enumerate(indices):
        logging.info("*" * 20)
        logging.info(f"** Fold {fold}/{args.num_folds - 1}")
        logging.info("*" * 20)

        logging.info("***** Building Model *****")
        model = __init_model()

        best_acc, best_speed_up = train(
            args=args,
            model=model,
            dataset=dataset,
            examples=examples,
            train_indices=train_indices,
            valid_indices=valid_indices,
            fold=fold
        )

        all_acc.append(best_acc)
        all_speed_up.append(best_speed_up)
        logging.info(f"Fold {fold}/{args.num_folds - 1} finished, "
                     f"acc: {best_acc}, speed_up: {best_speed_up}")

        torch.cuda.empty_cache()

    avg_acc = np.mean(all_acc)
    avg_speed_up = np.mean(all_speed_up)
    logging.info(f"Avg acc: {avg_acc}")
    logging.info(f"Avg speed up: {avg_speed_up}")
    wandb.log({"avg_acc": avg_acc, "avg_speed_up": avg_speed_up})


def train(args, model, dataset, examples, train_indices, valid_indices, fold):
    # --------------------------------------------------
    # train
    # --------------------------------------------------
    logging.info("***** Loading Datasets *****")
    train_dataset = Subset(dataset, indices=train_indices)
    logging.info(f"Train data size: {len(train_dataset)}")
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    valid_dataset = Subset(dataset, indices=valid_indices)
    valid_examples = [examples[idx] for idx in valid_indices]
    logging.info(f"Valid data size: {len(valid_dataset)}")
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    logging.info("***** Preparing Running Utils *****")
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
    best_model_dir = os.path.join(
        args.model_dir,
        f"fold_{fold}",
        f"best_{args.metric_for_best_model}",
    )
    early_stop = EarlyStopController(
        patience=args.patience,
        best_model_dir=best_model_dir,
        higher_is_better=True,
    )

    # label smoothing
    label_smoother = None
    if args.label_smoothing_factor != 0:
        label_smoother = LabelSmoother(epsilon=args.label_smoothing_factor)

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

                if label_smoother is not None:
                    labels = batch.pop("labels")
                else:
                    labels = None

                outputs = model(**batch)

                if label_smoother is not None:
                    loss = label_smoother(outputs, labels)
                else:
                    loss = outputs.loss

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
                        f"fold_{fold}/train/loss": loss.item(),
                        f"fold_{fold}/train/epoch": completed_steps
                                                    / num_update_steps_per_epoch,
                        f"fold_{fold}/train/learning_rate": lr_scheduler.optimizer.param_groups[0]["lr"]
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
            examples=valid_examples,
            fold=fold,
            epoch=epoch,
        )

        early_stop(
            score=valid_results[f"fold_{fold}/valid/{args.metric_for_best_model}"],
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

    model = early_stop.load_best_model()
    model.to(args.device)
    if args.num_devices > 1:
        model = torch.nn.DataParallel(model)

    valid_results = __eval(
        args=args,
        model=model,
        dataloader=valid_dataloader,
        examples=valid_examples,
        fold=fold,
        epoch=-1,
    )

    return valid_results[f"fold_{fold}/valid/acc"], valid_results[f"fold_{fold}/valid/speed_up"]


def __eval(args, model, dataloader, examples, fold, epoch):
    # statistics
    num_examples = 0
    num_steps = 0
    all_loss = []

    # used for computing metrics
    all_golds = []
    all_preds = []

    eval_bar = tqdm(dataloader, total=len(dataloader), ascii=True)
    model.eval()
    for step, batch in enumerate(eval_bar):
        for k, v in batch.items():
            batch[k] = v.to(args.device)
        labels = batch.get("labels")

        with torch.no_grad():
            outputs = model(**batch)

            loss = outputs.loss
            all_loss.append(loss.mean().item())

            logits = outputs.logits
            preds = np.argmax(logits.cpu().numpy(), axis=1)
            all_preds.extend([p.item() for p in preds])
            all_golds.extend(labels.squeeze(-1).cpu().numpy().tolist())

        num_examples += len(labels)
        num_steps += 1

    eval_loss = np.mean(all_loss)

    metric_prefix = f"fold_{fold}/valid"

    results = compute_acc(preds=all_preds, golds=all_golds, prefix=metric_prefix)
    if args.num_labels == 2:
        results.update(
            compute_p_r_f1(preds=all_preds, golds=all_golds, prefix=metric_prefix)
        )

    # compute speedup
    if args.task.startswith("device"):
        runtime_cpus = [example.runtime_cpu for example in examples]
        runtime_gpus = [example.runtime_gpu for example in examples]
        results.update(
            speed_up_for_device_mapping(
                platform=args.task.split("-")[-1],
                preds=all_preds,
                runtime_cpus=runtime_cpus,
                runtime_gpus=runtime_gpus,
                prefix=metric_prefix
            )
        )
    else:
        runtimes = [example.runtimes for example in examples]
        results.update(
            speed_up_for_thread_coarsening(
                preds=all_preds,
                runtimes=runtimes,
                prefix=metric_prefix
            )
        )

    results.update({f"{metric_prefix}/loss": eval_loss})
    wandb.log(results)

    results.update({
        f"{metric_prefix}/num_examples": num_examples,
        f"{metric_prefix}/num_steps": num_steps,
    })
    logging.info(results)

    # make sure that the metric for selecting best model is in the results when validating
    if f"{metric_prefix}/{args.metric_for_best_model}" not in results:
        raise ValueError(
            f"The metric for selecting best model is set to {args.metric_for_best_model}, "
            f"which is, however, not found in to validation results."
        )

    logging.info(f"Start gathering and saving results and details...")
    # save results to json file
    save_dir = os.path.join(args.eval_dir, f"fold_{fold}", f"epoch_{epoch}")
    os.makedirs(save_dir)
    with open(os.path.join(save_dir, "results.json"), mode="w", encoding="utf-8") as f:
        json.dump(results, f)

    # save summary
    inputs = [example.code for example in examples]
    labels = [example.label_txt for example in examples]
    df = pd.DataFrame(
        list(zip(inputs, labels, all_golds, all_preds)),
        columns=["Input", "Label", "Gold", "Pred"],
    )
    save_eval_details(save_dir=save_dir, detail_df=df)

    logging.info(f"Results and details are saved to {save_dir}")

    model.train()

    return results
