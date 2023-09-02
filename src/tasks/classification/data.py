from typing import Tuple, List, Optional
import logging
import os
import json
from tqdm import tqdm
from dataclasses import dataclass
from functools import partial

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from utils import load_pickle, save_pickle, multiprocess_func, read_file
from parser.input_utils import prepare_model_inputs


@dataclass
class Example(object):
    code: str
    label_idx: int
    label_txt: str


@dataclass
class InputFeature(object):
    input_ids: torch.Tensor  # [T]
    all_bb_encoder_input_ids: torch.Tensor  # [max_num_bbs, T]
    cfg_matrix: torch.Tensor  # [num_bbs, num_bbs]
    dfg_matrix: torch.Tensor  # [num_vars, num_vars]
    bb_var_matrix: torch.Tensor  # [num_bbs, num_vars]
    labels: torch.Tensor


class ClassificationDataset(Dataset):
    def __init__(self, features):
        super().__init__()
        self.features = features

    def __getitem__(self, item):
        # return_dict = self.features[item].__dict__
        # print({k: v.size() for k, v in return_dict.items()})
        return self.features[item].__dict__

    def __len__(self):
        return len(self.features)


def prepare_dataset(
    args, tokenizer, cfg_vocab, dfg_vocab, split
) -> Tuple[ClassificationDataset, List[Example]]:
    # check split
    assert split in ["train", "valid", "test"]
    dataset_dir = os.path.join(args.dataset_root, "POJ-104")
    logging.info(f"Start preparing {split} data...")

    # load examples
    dataset_cache_path = os.path.join(
        dataset_dir, "cache", f"{args.task}-{split}-dataset.pk"
    )
    examples_cache_path = os.path.join(
        dataset_dir, "cache", f"{args.task}-{split}-examples.pk"
    )
    if args.use_data_cache and os.path.exists(examples_cache_path):
        # load parsed examples from cache
        logging.info(f"Find cache of examples: {examples_cache_path}")
        examples = load_pickle(examples_cache_path)
        assert isinstance(examples, list)
        assert isinstance(examples[0], Example)
        logging.info(f"Examples are loaded from cache")
    else:
        # load and parse examples from disk files
        logging.info("Start loading split metadata...")
        with open(
            os.path.join(dataset_dir, f"{split}.jsonl"), mode="r", encoding="utf-8"
        ) as f:
            lines = f.readlines()

        code_dir = os.path.join(dataset_dir, "optimized")
        logging.info("Start loading data...")
        examples = []
        # lines = lines[:20]
        for line in tqdm(lines, ascii=True):
            data = json.loads(line)
            code_path = os.path.join(
                code_dir, data["label"], f"{data['solution_id']}-Os-ffast-math.ll"
            )
            if os.path.exists(code_path):
                code = read_file(code_path)
                examples.append(
                    Example(
                        code=code,
                        label_idx=int(data["label"]) - 1,
                        label_txt=data["label"],
                    )
                )
        # save as cache if needed
        if args.use_data_cache:
            save_pickle(obj=examples, path=examples_cache_path)

    # load dataset
    if args.use_data_cache and os.path.exists(dataset_cache_path):
        # load parsed dataset from cache
        logging.info(f"Find cache of dataset: {dataset_cache_path}")
        dataset = load_pickle(dataset_cache_path)
        assert isinstance(dataset, ClassificationDataset)
        logging.info(f"Dataset is loaded from cache")
    else:
        # build dataset from examples
        logging.info("Start parsing and tokenizing...")
        features = convert_examples_to_features(
            args=args,
            examples=examples,
            tokenizer=tokenizer,
            cfg_vocab=cfg_vocab,
            dfg_vocab=dfg_vocab,
        )

        dataset = ClassificationDataset(features)

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
        labels=torch.tensor([example.label_idx], dtype=torch.long)
    )

    return input_feature


def convert_examples_to_features(
    args,
    examples: List[Example],
    tokenizer: PreTrainedTokenizerFast,
    cfg_vocab: dict,
    dfg_vocab: dict,
) -> List[Optional[InputFeature]]:
    encode_func = partial(
        encode_example,
        args=args,
        tokenizer=tokenizer,
        cfg_vocab=cfg_vocab,
        dfg_vocab=dfg_vocab,
    )

    features = multiprocess_func(encode_func, examples, single_thread=args.single_thread_parsing)
    features = [feature for feature in features if feature is not None]

    return features
