import os
import logging
import json
from dataclasses import dataclass
from typing import Tuple, List
from functools import partial
import torch
from transformers import PreTrainedTokenizerFast
from torch.utils.data import Dataset

from utils import (
    save_pickle,
    load_pickle,
    multiprocess_func,
    read_file,
)
from parser.input_utils import prepare_model_inputs


@dataclass
class DeviceMappingExample:
    code: str
    label_idx: int
    label_txt: str
    input_size: int
    wg_size: int
    benchmark: str
    runtime_cpu: float
    runtime_gpu: float


@dataclass
class ThreadCoarseningExample:
    code: str
    label_idx: int  # 0, 1, 2, 3, 4, 5
    label_txt: str  # 1, 2, 4, 8, 16, 32
    kernel: str  # benchmark
    platform: str  # platform
    runtimes: dict


@dataclass
class InputFeature(object):
    input_ids: torch.Tensor  # [T]
    all_bb_encoder_input_ids: torch.Tensor  # [max_num_bbs, T]
    cfg_matrix: torch.Tensor  # [num_bbs, num_bbs]
    dfg_matrix: torch.Tensor  # [num_vars, num_vars]
    bb_var_matrix: torch.Tensor  # [num_bbs, num_vars]
    labels: torch.Tensor


class KFoldClassificationDataset(Dataset):
    def __init__(self, features):
        super().__init__()
        self.features = features

    def __getitem__(self, item):
        return self.features[item].__dict__

    def __len__(self):
        return len(self.features)


def prepare_dataset(args, tokenizer, cfg_vocab, dfg_vocab):
    logging.info(f"Start preparing data...")
    assert args.task in [
        "device-nvidia",
        "device-amd",
        "thread-fermi",
        "thread-kepler",
        "thread-cypress",
        "thread-tahiti",
    ]
    if args.task in ["device-nvidia", "device-amd"]:
        return prepare_device_data(args, tokenizer, cfg_vocab, dfg_vocab)
    elif args.task in [
        "thread-fermi",
        "thread-kepler",
        "thread-cypress",
        "thread-tahiti",
    ]:
        return prepare_thread_data(args, tokenizer, cfg_vocab, dfg_vocab)
    else:
        raise ValueError(f"Task `{args.task}` not supported.")


def prepare_device_data(
    args, tokenizer, cfg_vocab, dfg_vocab
) -> Tuple[KFoldClassificationDataset, List[DeviceMappingExample]]:
    dataset_dir = os.path.join(args.dataset_root, "DeviceMapping")

    examples_cache_path = os.path.join(dataset_dir, "cache", f"{args.task}-examples.pk")
    dataset_cache_path = os.path.join(dataset_dir, "cache", f"{args.task}-dataset.pk")
    if args.use_data_cache and os.path.exists(examples_cache_path):
        # load parsed examples from cache
        logging.info(f"Find cache of examples: {examples_cache_path}")
        examples = load_pickle(examples_cache_path)
        assert isinstance(examples, list)
        assert isinstance(examples[0], DeviceMappingExample)
        logging.info("Examples are loaded from cache")
    else:
        logging.info("Start loading and parsing data...")
        platform = args.task.split("-")[-1]
        with open(
            os.path.join(dataset_dir, f"{platform}.jsonl"), mode="r", encoding="utf-8"
        ) as f:
            lines = f.readlines()

        ir_dir = os.path.join(dataset_dir, "optimized")
        examples = []
        for line in lines:
            data = json.loads(line)
            filename = data["file_name"]
            input_size = int(data["transfer"])
            wg_size = int(data["wgsize"])
            oracle = data["oracle"]
            if oracle == "GPU":
                label_txt = platform
                label = 1
            else:
                label_txt = "cpu"
                label = 0
            ir_path = os.path.join(ir_dir, f"{filename}-Os-ffast-math.ll")
            if os.path.exists(ir_path):
                ir = read_file(ir_path)
                examples.append(
                    DeviceMappingExample(
                        code=ir,
                        input_size=input_size,
                        wg_size=wg_size,
                        label_idx=label,
                        label_txt=label_txt,
                        benchmark=data["benchmark"],
                        runtime_cpu=float(data["runtime_cpu"]),
                        runtime_gpu=float(data["runtime_gpu"]),
                    )
                )
        # save as cache if needed
        if args.use_data_cache:
            save_pickle(obj=examples, path=examples_cache_path)

    # load dataset
    if args.use_data_cache and os.path.exists(dataset_cache_path):
        # load parsed data from cache
        logging.info(f"Find cache of dataset: {dataset_cache_path}")
        dataset = load_pickle(dataset_cache_path)
        assert isinstance(dataset, KFoldClassificationDataset)
        logging.info(f"Dataset is loaded from cache")
    else:
        # build dataset from examples
        logging.info("Start tokenizing...")
        features = convert_examples_to_features(args, examples, tokenizer, cfg_vocab, dfg_vocab)

        dataset = KFoldClassificationDataset(features)

        if args.use_data_cache:
            save_pickle(obj=dataset, path=dataset_cache_path)

    return dataset, examples


def prepare_thread_data(
    args, tokenizer, cfg_vocab, dfg_vocab
) -> Tuple[KFoldClassificationDataset, List[ThreadCoarseningExample]]:
    dataset_dir = os.path.join(args.dataset_root, "ThreadCoarsening")
    platform = args.task.split("-")[-1].capitalize()

    examples_cache_path = os.path.join(dataset_dir, "cache", f"{args.task}-{args.use_all_options}-examples.pk")
    dataset_cache_path = os.path.join(dataset_dir, "cache", f"{args.task}-{args.use_all_options}-dataset.pk")
    if args.use_data_cache and os.path.exists(examples_cache_path):
        # load parsed examples from cache
        logging.info(f"Find cache of examples: {examples_cache_path}")
        examples = load_pickle(examples_cache_path)
        assert isinstance(examples, list)
        assert isinstance(examples[0], ThreadCoarseningExample)
        logging.info("Examples are loaded from cache")
    else:
        logging.info("Start loading and parsing data...")
        with open(
            os.path.join(dataset_dir, f"{platform}.jsonl"), mode="r", encoding="utf-8"
        ) as f:
            lines = f.readlines()

        ir_dir = os.path.join(dataset_dir, "optimized")
        examples = []
        label_list = [1, 2, 4, 8, 16, 32]
        for line in lines:
            data = json.loads(line)
            filename = data["kernel"]
            oracle = int(data["oracle"])
            label_txt = str(oracle)
            label_int = int(label_txt)
            if label_int not in label_list:
                label_list.append(label_int)
            label = label_list.index(label_int)
            runtimes = {}
            for key, value in data.items():
                if key.startswith("runtime_"):
                    cf = key.split("_")[-1]
                    runtimes[int(cf)] = value

            if args.use_all_options:
                options_list = [
                    ["-O1"],
                    ["-O1", "-ffast-math"],
                    ["-O2"],
                    ["-O2", "-ffast-math"],
                    ["-O3"],
                    ["-O3", "-ffast-math"],
                    ["-Oz"],
                    ["-Oz", "-ffast-math"],
                    ["-Os"],
                    ["-Os", "-ffast-math"],
                ]
            else:
                options_list = [
                    ["-Os", "-ffast-math"],
                ]
            for options in options_list:
                option_str = "".join(options)
                ir_path = os.path.join(ir_dir, f"{filename}{option_str}.ll")
                ir = read_file(ir_path)
                examples.append(
                    ThreadCoarseningExample(
                        code=ir,
                        label_idx=label,
                        label_txt=label_txt,
                        kernel=filename,
                        platform=platform,
                        runtimes=runtimes,
                    )
                )
        # save as cache if needed
        if args.use_data_cache:
            save_pickle(obj=examples, path=examples_cache_path)

    # load dataset
    if args.use_data_cache and os.path.exists(dataset_cache_path):
        # load parsed data from cache
        logging.info(f"Find cache of dataset: {dataset_cache_path}")
        dataset = load_pickle(dataset_cache_path)
        assert isinstance(dataset, KFoldClassificationDataset)
        logging.info(f"Dataset is loaded from cache")
    else:
        # build dataset from examples
        logging.info("Start tokenizing...")
        features = convert_examples_to_features(args, examples, tokenizer, cfg_vocab,  dfg_vocab)

        dataset = KFoldClassificationDataset(features)

        if args.use_data_cache:
            save_pickle(obj=dataset, path=dataset_cache_path)

    return dataset, examples


def encode_example(example, args, tokenizer, cfg_vocab, dfg_vocab):
    if args.task.startswith("device"):
        extra_tokens = [str(example.input_size), str(example.wg_size)]
    else:
        extra_tokens = None
    model_inputs = prepare_model_inputs(
        args=args,
        ir_content=example.code,
        tokenizer=tokenizer,
        cfg_vocab=cfg_vocab,
        dfg_vocab=dfg_vocab,
        extra_tokens=extra_tokens
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
    args, examples, tokenizer: PreTrainedTokenizerFast, cfg_vocab, dfg_vocab
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
