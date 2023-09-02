import logging
import os
from dataclasses import dataclass
from typing import List
from tqdm import tqdm
from functools import partial
import random

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from utils import load_pickle, save_pickle, multiprocess_func, read_file
from parser.input_utils import prepare_model_inputs


@dataclass
class Example:
    code: str
    idx: str        # unique index for retrieval
    label: int      # identify the code clone


@dataclass
class InputFeature:
    input_ids: torch.Tensor  # [T]
    all_bb_encoder_input_ids: torch.Tensor  # [max_num_bbs, T]
    cfg_matrix: torch.Tensor  # [num_bbs, num_bbs]
    dfg_matrix: torch.Tensor  # [num_vars, num_vars]
    bb_var_matrix: torch.Tensor  # [num_bbs, num_vars]
    labels: torch.Tensor


class RetrievalDataset(Dataset):
    def __init__(self, features: List[InputFeature]):
        self.features = features
        self.size = len(self.features)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        feature = self.features[i]
        cur_label = feature.labels

        pos_feature = None
        neg_feature = None
        while pos_feature is None or neg_feature is None:
            random_idx = random.randint(0, len(self.features) - 1)
            if random_idx != i:
                if pos_feature is None and self.features[random_idx].labels == cur_label:
                    pos_feature = self.features[random_idx]
                elif neg_feature is None and self.features[random_idx].labels != cur_label:
                    neg_feature = self.features[random_idx]

        model_inputs = feature.__dict__.copy()
        model_inputs.update({
            "pos_input_ids": pos_feature.input_ids,
            "pos_all_bb_encoder_input_ids": pos_feature.all_bb_encoder_input_ids,
            "pos_cfg_matrix": pos_feature.cfg_matrix,
            "pos_dfg_matrix": pos_feature.dfg_matrix,
            "pos_bb_var_matrix": pos_feature.bb_var_matrix,

            "neg_input_ids": neg_feature.input_ids,
            "neg_all_bb_encoder_input_ids": neg_feature.all_bb_encoder_input_ids,
            "neg_cfg_matrix": neg_feature.cfg_matrix,
            "neg_dfg_matrix": neg_feature.dfg_matrix,
            "neg_bb_var_matrix": neg_feature.bb_var_matrix,
        })

        return model_inputs


def prepare_dataset(
        args, tokenizer, cfg_vocab, dfg_vocab, split
) -> (RetrievalDataset, List[Example]):
    assert split in ["train", "valid", "test"]
    dataset_dir = os.path.join(args.dataset_root, "POJ-104")
    logging.info(f"Start preparing {split} data from {dataset_dir}")

    # load examples
    dataset_cache_path = os.path.join(dataset_dir, "cache", f"{args.task}-{split}-dataset.pk")
    examples_cache_path = os.path.join(dataset_dir, "cache", f"{args.task}-{split}-examples.pk")
    if args.use_data_cache and os.path.exists(examples_cache_path):
        # load parsed examples from cache
        logging.info(f"Find cache of examples: {examples_cache_path}")
        examples = load_pickle(examples_cache_path)
        assert isinstance(examples, list)
        assert isinstance(examples[0], Example)
        logging.info("Examples are loaded from cache")
    else:
        # load and parse examples from disk files
        logging.info("Start loading and parsing data...")
        if split == "train":
            labels = range(1, 65)
        elif split == "valid":
            labels = range(65, 81)
        else:
            labels = range(81, 105)

        examples = []
        for label in tqdm(labels, ascii=True):
            label_dir = os.path.join(dataset_dir, "optimized", str(label))
            for filename in os.listdir(label_dir):
                if filename.endswith(".ll"):
                    code = read_file(os.path.join(label_dir, filename))
                    examples.append(
                        Example(code=code, idx=f"{label}/{filename}", label=int(label))
                    )
        # save as cache if needed
        if args.use_data_cache:
            save_pickle(obj=examples, path=examples_cache_path)

    # load dataset
    if args.use_data_cache and os.path.exists(dataset_cache_path):
        # load parsed data from cache
        logging.info(f"Find cache of dataset: {dataset_cache_path}")
        dataset = load_pickle(dataset_cache_path)
        assert isinstance(dataset, RetrievalDataset)
        logging.info(f"Dataset is loaded from cache")
    else:
        # build dataset from examples
        logging.info("Start tokenizing...")
        features = convert_examples_to_features(args, examples, tokenizer, cfg_vocab, dfg_vocab)
        dataset = RetrievalDataset(features)

        if args.use_data_cache:
            save_pickle(obj=dataset, path=dataset_cache_path)

    return dataset, examples


def encode_example(example, args, tokenizer, cfg_vocab, dfg_vocab):
    model_inputs = prepare_model_inputs(
        args=args,
        ir_content=example.code,
        tokenizer=tokenizer,
        cfg_vocab=cfg_vocab,
        dfg_vocab=dfg_vocab,
    )
    if model_inputs is None:
        return None

    input_feature = InputFeature(
        input_ids=torch.tensor(model_inputs["input_ids"], dtype=torch.int),
        all_bb_encoder_input_ids=torch.tensor(model_inputs["all_bb_encoder_input_ids"], dtype=torch.int),
        cfg_matrix=torch.tensor(model_inputs["cfg_matrix"], dtype=torch.int),
        dfg_matrix=torch.tensor(model_inputs["dfg_matrix"], dtype=torch.int),
        bb_var_matrix=torch.tensor(model_inputs["bb_var_matrix"], dtype=torch.int),
        labels=torch.tensor(example.label, dtype=torch.int)
    )

    return input_feature


def convert_examples_to_features(
        args,
        examples: List[Example],
        tokenizer: PreTrainedTokenizerFast,
        cfg_vocab: dict,
        dfg_vocab: dict
) -> (List[InputFeature], int):

    encode_func = partial(
        encode_example,
        args=args,
        tokenizer=tokenizer,
        cfg_vocab=cfg_vocab,
        dfg_vocab=dfg_vocab
    )

    features = multiprocess_func(encode_func, examples, single_thread=args.single_thread_parsing)
    features = [feature for feature in features if feature is not None]

    return features
