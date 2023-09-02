import os
import logging
import pickle
from prettytable import PrettyTable
import multiprocessing
from tqdm import tqdm
from typing import List
from dataclasses import dataclass
import json

import torch
import torch.nn as nn
from transformers import (
    RobertaTokenizerFast,
    AutoModel,
    AutoConfig,
)

from fair import FairForSequenceClassification, FairForRetrieval


# https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/trainer_pt_utils.py#L473
@dataclass
class LabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (`float`, *optional*, defaults to 0.1):
            The label smoothing factor.
        ignore_index (`int`, *optional*, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(self, model_output, labels, shift_labels=False):
        logits = (
            model_output["logits"]
            if isinstance(model_output, dict)
            else model_output[0]
        )
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (
            num_active_elements * log_probs.shape[-1]
        )
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss


class EarlyStopController(object):
    """
    A controller for early stopping.
    Args:
        patience (int):
            Maximum number of consecutive epochs without breaking the best record.
        higher_is_better (bool, optional, defaults to True):
            Whether a higher record is seen as a better one.
    """

    def __init__(self, patience: int, best_model_dir, higher_is_better=True):
        self.patience = patience
        self.best_model_dir = best_model_dir
        self.higher_is_better = higher_is_better

        self.early_stop = False
        self.hit = False
        self.counter = 0

        self.best_score = None
        self.best_state = {"epoch": None, "step": None}

    def __call__(self, score: float, model, epoch: int, step: int):
        """Calls this after getting the validation metric each epoch."""
        # first calls
        if self.best_score is None:
            self.__update_best(score, model, epoch, step)
        else:
            # not hits the best record
            if (self.higher_is_better and score < self.best_score) or (
                not self.higher_is_better and score > self.best_score
            ):
                self.hit = False
                self.counter += 1
                if self.counter > self.patience:
                    self.early_stop = True
            # hits the best record
            else:
                self.__update_best(score, model, epoch, step)

    def __update_best(self, score, model, epoch, step):
        self.best_score = score
        self.hit = True
        self.counter = 0
        self.best_state["epoch"] = epoch
        self.best_state["step"] = step

        os.makedirs(self.best_model_dir, exist_ok=True)
        torch.save(
            model.module if isinstance(model, torch.nn.DataParallel) else model,
            os.path.join(self.best_model_dir, "model.pt")
        )
        # save fair model
        if hasattr(model, "fair"):
            model.fair.save_pretrained(
                os.path.join(self.best_model_dir, "FairModel")
            )

    def load_best_model(self):
        load_path = os.path.join(self.best_model_dir, "model.pt")
        obj = None
        if os.path.exists(load_path):
            obj = torch.load(load_path, map_location=torch.device("cpu"))
        return obj

    def get_best_score(self):
        return self.best_score


class MixedPrecisionManager(object):
    def __init__(self, activated):
        self.activated = activated

        if self.activated:
            self.scaler = torch.cuda.amp.GradScaler()

    def context(self):
        return torch.cuda.amp.autocast() if self.activated else NullContextManager()

    def backward(self, loss):
        if self.activated:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, model, optimizer, max_grad_norm=None):
        if self.activated:
            if max_grad_norm is not None:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()


class NullContextManager(object):
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource

    def __enter__(self):
        return self.dummy_resource

    def __exit__(self, *args):
        pass


def load_tokenizer(args):
    logging.info("Start loading tokenizer and vocabs...")

    tokenizer = RobertaTokenizerFast.from_pretrained(
        os.path.join(args.vocab_root, args.tokenizer_dir)
    )
    # tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")
    args.vocab_size = tokenizer.vocab_size

    args.bos_token_id = tokenizer.bos_token_id
    args.pad_token_id = tokenizer.pad_token_id
    args.eos_token_id = tokenizer.eos_token_id
    args.mask_token_id = tokenizer.mask_token_id

    args.bos_token = tokenizer.bos_token
    args.pad_token = tokenizer.pad_token
    args.eos_token = tokenizer.eos_token
    args.mask_token = tokenizer.mask_token

    logging.info(f"Tokenizer loaded, size: {args.vocab_size}")

    with open(os.path.join(args.vocab_root, args.cfg_vocab_name), mode="r", encoding="utf-8") as f:
        cfg_vocab = json.load(f)
    args.cfg_vocab_size = len(cfg_vocab)
    logging.info(f"CFG flow type vocabulary loaded, size: {args.cfg_vocab_size}")

    with open(os.path.join(args.vocab_root, args.dfg_vocab_name), mode="r", encoding="utf-8") as f:
        dfg_vocab = json.load(f)
    args.dfg_vocab_size = len(dfg_vocab)
    logging.info(f"DFG flow type vocabulary loaded, size: {args.dfg_vocab_size}")

    return tokenizer, cfg_vocab, dfg_vocab


def model_summary(model):
    if hasattr(model, "config"):
        logging.debug(f"Model configuration:\n{model.config}")
    logging.info(f"Model type: {model.__class__.__name__}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Trainable parameters: {human_format(num_params)}")
    logging.debug(f"Layer-wise parameters:\n{layer_wise_parameters(model)}")


def human_format(num):
    """Transfer number into a readable format."""
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), ["", "K", "M", "B", "T"][magnitude]
    )


def layer_wise_parameters(model):
    """Returns a printable table representing the layer-wise model parameters, their shapes and numbers"""
    table = PrettyTable()
    table.field_names = ["Layer Name", "Output Shape", "Param #"]
    table.align["Layer Name"] = "l"
    table.align["Output Shape"] = "r"
    table.align["Param #"] = "r"
    for name, parameters in model.named_parameters():
        if parameters.requires_grad:
            table.add_row([name, str(list(parameters.shape)), parameters.numel()])
    return table


def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode="wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    if not os.path.exists(path):
        raise ValueError(f"Pickle file not exists: {os.path.abspath(path)}")
    with open(path, mode="rb") as f:
        obj = pickle.load(f)
    return obj


def multiprocess_func(func, items, num_processors=None, single_thread=False) -> List:
    processes = num_processors if num_processors else multiprocessing.cpu_count()
    if processes > 1 and not single_thread:
        with multiprocessing.Pool(processes=processes) as p:
            results = list(p.map(func, tqdm(items, total=len(items), ascii=True)))
    else:
        results = [func(item) for item in tqdm(items, total=len(items), ascii=True)]
    return results


def read_file(path, encoding="utf-8"):
    with open(path, mode="r", encoding=encoding) as f:
        content = f.read()
    return content


def remove_special_tokens(s, tokenizer) -> str:
    """Removes all special tokens from given str."""
    for token in tokenizer.all_special_tokens:
        s = s.replace(token, " ")
    return s


def tokenize(txt, tokenizer, max_length) -> List[int]:
    """Converts code to input ids, only for code."""
    source = remove_special_tokens(txt, tokenizer)
    return tokenizer.encode(
        source, padding="max_length", max_length=max_length, truncation=True
    )


def save_eval_details(save_dir, detail_df, filename_prefix=None):
    os.makedirs(save_dir, exist_ok=True)
    # save as csv
    detail_df.to_csv(
        os.path.join(
            save_dir,
            f"{filename_prefix}_details.csv"
            if filename_prefix is not None
            else "summary.csv",
        ),
        encoding="utf-8",
    )
    # save as json
    detail_df.to_json(
        os.path.join(
            save_dir,
            f"{filename_prefix}_details.json"
            if filename_prefix is not None
            else "summary.json",
        ),
        orient="records",
        indent=4,
    )
    # save as html table
    html_table = detail_df.to_html()
    with open(
        os.path.join(
            save_dir,
            f"{filename_prefix}_details.html"
            if filename_prefix is not None
            else "summary.html",
        ),
        mode="w",
        encoding="utf-8",
    ) as f:
        f.write(html_table)
