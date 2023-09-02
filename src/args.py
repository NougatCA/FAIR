from argparse import ArgumentParser


def add_args(parser: ArgumentParser):
    # data and saving
    parser.add_argument("--dataset_root", type=str, default="../../datasets/",
                        help="The root of the datasets.")
    parser.add_argument("--use_data_cache", default=True, action="store_true",
                        help="Use/Save example/dataset pickle cache.")
    parser.add_argument("--save_valid_details", default=False, action="store_true",
                        help="Save validation summary.")
    parser.add_argument("--use_all_options", type=bool, default=False,
                        help="Whether to use all compilation options to argument data.")

    # model
    parser.add_argument("--model_size", type=str, default="base",
                        choices=["small", "base"],
                        help="The size version of model, only matters when training from scratch.")
    parser.add_argument("--model_path", type=str, default="../models/pre-trained/",
                        help="Path of the model checkpoint.")
    # parser.add_argument("--train_from_scratch", default=False, action="store_true",
    #                     help="Train model from scratch.")

    # task
    parser.add_argument("--task", type=str, default="thread-fermi",
                        choices=["pre-train",
                                 "cls-poj104",
                                 "device-nvidia", "device-amd",
                                 "thread-fermi", "thread-kepler", "thread-cypress", "thread-tahiti",
                                 "retrieval-poj104", "retrieval-gcj"],
                        help="The task to run.")
    parser.add_argument("--num_labels", type=int, default=None,
                        help="The total number of labels.")
    parser.add_argument("--only_test", default=False, action="store_true",
                        help="Only perform testing.")

    # vocabs
    parser.add_argument("--vocab_root", type=str, default="../vocabs/",
                        help="The root of the vocabulary, e.g., tokenizer, flow type vocabs.")
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer",
                        help="The directory name of tokenizer.")
    parser.add_argument("--cfg_vocab_name", type=str, default="cfg_type_vocab.json",
                        help="The json file name of the control flow type vocabulary.")
    parser.add_argument("--dfg_vocab_name", type=str, default="dfg_type_vocab.json",
                        help="The json file name of the data flow type vocabulary.")

    # hyper-parameter
    # runtime
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size, note that this is the total batch size, "
                             "instead of the batch size per device.")
    parser.add_argument("--eval_batch_size", type=int, default=None,
                        help="Validation and testing batch size, default to training batch size.")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="The total number of training epochs.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="The total training steps, default to None, overrides `--num_epochs`. "
                             "Each step means one optimization step, "
                             "i.e., includes gradient accumulation.")
    parser.add_argument("--warmup_steps", type=float, default=0.1,
                        help="Number of warmup steps of learning rate.")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Logging intervals.")
    # optimizer
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay of optimizer.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm, None to disable.")
    # loss
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="The number of gradient accumulation steps.")
    parser.add_argument("--label_smoothing_factor", type=float, default=0.1,
                        help="Label smoothing factor, 0 to disable.")
    # early stopping
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience.")
    # limitation
    parser.add_argument("--max_input_length", type=int, default=510,
                        help="Max input length.")
    parser.add_argument("--max_bb_input_length", type=int, default=254,
                        help="Max input length of bb encoder.")
    parser.add_argument("--max_num_bbs", type=int, default=64,
                        help="Max number of bbs for an instance.")
    parser.add_argument("--max_var_input_length", type=int, default=256,
                        help="Maximum length of the variable input sequence.")
    parser.add_argument("--max_extra_input_length", type=int, default=16,
                        help="Maximum length of the variable input sequence.")

    # run
    parser.add_argument("--run_name", type=str, default=None,
                        help="The name of this run.")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="The random seed.")
    parser.add_argument("--cuda_visible_devices", type=str, default=None,
                        help="Index (Indices) of the GPU to use in a cluster.")
    parser.add_argument("--no_cuda", default=False, action="store_true",
                        help="Disable cuda, overrides cuda_visible_devices.")
    parser.add_argument("--fp16", type=bool, default=True,
                        help="Disable fp16 mixed precision.")
    parser.add_argument("--wandb_mode", type=str, default="offline",
                        choices=["online", "offline", "disabled"],
                        help="Set the wandb mode.")
    # BB encoder
    # parser.add_argument(
    #     "--bb_encoder_num_heads",
    #     type=int,
    #     default=12,
    #     help="The number of self-attention heads of BB encoder."
    # )
    # parser.add_argument(
    #     "--bb_encoder_num_layers",
    #     type=int,
    #     default=6,
    #     help="The number of layers of BB encoder."
    # )

    # max lengths
    # parser.add_argument(
    #     "--max_bb_length",
    #     type=int,
    #     default=256,
    #     help="Maximum sequence length of a basic block."
    # )
    # parser.add_argument(
    #     "--max_bb_num",
    #     type=int,
    #     default=49,
    #     help="Maximum number of basic blocks."
    # )
    # parser.add_argument(
    #     "--max_var_length",
    #     type=int,
    #     default=190,
    #     help="Maximum length of the variable input sequence."
    # )
    # parser.add_argument(
    #     "--max_extra_length",
    #     type=int,
    #     default=15,
    #     help="Maximum length of the variable input sequence."
    # )

    # features
    # parser.add_argument(
    #     "--add_flow_attn_scores",
    #     type=int,
    #     default=True,
    #     help="Whether to add flow type self-attention scores."
    # )

    # runtime
    parser.add_argument("--single_thread_parsing", default=True, action="store_true",
                        help="Use single process to parse and tokenize examples.")


def check_args(args):
    """Check if args values are valid, and conduct some default settings."""
    # assert args.task is not None, "Please specific a task to run."
    # assert os.path.exists(args.dataset_root), f"Dataset root \"{args.dataset_root}\" not exists."
    # assert os.path.exists(args.vocab_root), f"Vocabulary root \"{args.vocab_root}\" not exists."
    # assert os.path.exists(os.path.join(args.vocab_root, args.tokenizer_dir)), \
    #     f"Tokenizer directory \"{os.path.join(args.vocab_root, args.tokenizer_name)}\" not exists."
    # assert os.path.exists(os.path.join(args.vocab_root, args.cfg_vocab_name)), \
    #     f"CFG Flow type json file \"{os.path.join(args.vocab_root, args.cfg_vocab)}\" not exists."
    # assert os.path.exists(os.path.join(args.vocab_root, args.dfg_vocab_name)), \
    #     f"DFG Flow type json file \"{os.path.join(args.vocab_root, args.dfg_vocab)}\" not exists."

    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size

    if args.num_epochs is None and args.num_train_steps is None:
        raise ValueError("One or both of `--num_train_epochs` or `--num_train_steps` must be specified.")

    # task-specific setting
    if args.task == "cls-poj104":
        args.num_labels = 104
        args.metric_for_best_model = "acc"
    if args.task.startswith("retrieval"):
        args.metric_for_best_model = "map"
    if args.task.startswith("device") or args.task.startswith("thread"):
        args.num_folds = 5
        args.metric_for_best_model = "speed_up"
        args.num_labels = 2 if args.task.startswith("device") else 6
