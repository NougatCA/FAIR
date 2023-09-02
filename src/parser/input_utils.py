import numpy as np

from .parser import parse_ir_file
# from .parser_new import parse_ir_file

def prepare_model_inputs(args, ir_content: str, tokenizer, cfg_vocab, dfg_vocab, extra_tokens=None):
    ir = parse_ir_file(
        content=ir_content,
        cfg_vocab=cfg_vocab,
        dfg_vocab=dfg_vocab
    )
    if ir is None:
        return None
    # generate input sequences for bb encoder
    bb_input_strings = ir.generate_bb_input_strings(max_num_bbs=args.max_num_bbs)
    all_bb_encoder_input_ids = [
        tokenizer.encode(
            string,
            padding="max_length",
            truncation=True,
            max_length=args.max_bb_input_length
        )
        for string in bb_input_strings
    ]
    num_bbs = len(bb_input_strings)
    bb_padding_length = args.max_num_bbs - num_bbs
    if bb_padding_length > 0:
        all_bb_encoder_input_ids.extend([
            [-1 for _ in range(args.max_bb_input_length)] for _ in range(bb_padding_length)
        ])
    # placeholder for bb part of encoder
    bb_input_ids = (
            [tokenizer.mask_token_id] * num_bbs
            + [tokenizer.pad_token_id] * bb_padding_length
    )

    tokenizer.add_prefix_space = True
    var_input_tokens = [var[:20] for var in ir.var_list]
    var_encoded = tokenizer(
        var_input_tokens,
        is_split_into_words=True,
        padding="max_length",
        truncation=True,
        max_length=args.max_var_input_length,
        add_special_tokens=False,
    )
    # [num_var_tokens]
    var_input_ids = var_encoded["input_ids"]
    word_ids = [idx for idx in var_encoded.word_ids() if idx is not None]
    # expand matrix due to the tokenization
    expanded_dfg_matrix = expand_dfg_matrix(dfg_matrix=ir.dfg_matrix, word_ids=word_ids)
    expanded_bb_var_matrix = expand_bb_var_matrix(
        bb_var_matrix=ir.bb_var_matrix, word_ids=word_ids
    )

    # extra tokens
    if extra_tokens is not None:
        extra_tokens = f" {tokenizer.eos_token} ".join(extra_tokens).split() + [
            tokenizer.eos_token
        ]
        extra_input_ids = tokenizer.encode(
            extra_tokens,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=args.max_extra_input_length,
            add_special_tokens=False,
        )
    else:
        extra_input_ids = [tokenizer.pad_token_id] * args.max_extra_input_length

    input_ids = (
            [tokenizer.bos_token_id]  # 1
            + bb_input_ids  # max_bb_input_length
            + [tokenizer.eos_token_id] * 2  # 2
            + var_input_ids  # max_var_input_length
            + [tokenizer.eos_token_id] * 2  # 2
            + extra_input_ids  # max_extra_input_length
    )

    padded_cfg = pad_or_truncate_matrix(
        ir.cfg_matrix + 1, target_size=(args.max_num_bbs, args.max_num_bbs)
    )
    padded_dfg = pad_or_truncate_matrix(
        expanded_dfg_matrix + 1, target_size=(args.max_var_input_length, args.max_var_input_length)
    )
    padded_bb_var = pad_or_truncate_matrix(
        expanded_bb_var_matrix, target_size=(args.max_num_bbs, args.max_var_input_length)
    )

    return {
        "all_bb_encoder_input_ids": all_bb_encoder_input_ids,   # [max_num_bbs, max_bb_input_length]
        "input_ids": input_ids,                                 # [max_input_length]
        "cfg_matrix": padded_cfg,                               # [max_num_bbs, max_num_bbs]
        "dfg_matrix": padded_dfg,                               # [max_num_vars, max_num_vars]
        "bb_var_matrix": padded_bb_var                          # [max_num_bbs, max_num_vars]
    }


def expand_dfg_matrix(dfg_matrix, word_ids):
    num_tokens = len(word_ids)
    new_dfg_matrix = np.full((num_tokens, num_tokens), fill_value=-1, dtype=np.int8)
    for i in range(num_tokens):
        for j in range(num_tokens):
            new_dfg_matrix[i][j] = dfg_matrix[word_ids[i]][word_ids[j]]
    return new_dfg_matrix


def expand_bb_var_matrix(bb_var_matrix, word_ids):
    num_tokens = len(word_ids)
    num_bbs = len(bb_var_matrix)
    new_bb_var_matrix = np.full((num_bbs, num_tokens), fill_value=0, dtype=bool)
    for i in range(num_bbs):
        for j in range(1, num_tokens):
            new_bb_var_matrix[i][j] = bb_var_matrix[i][word_ids[j]]
    return new_bb_var_matrix


def pad_or_truncate_matrix(matrix, target_size, padding_id=0):
    padded = np.full(shape=target_size, fill_value=padding_id, dtype=np.uint8)
    padded[: matrix.shape[0], : matrix.shape[1]] = matrix[:target_size[0], :target_size[1]]
    return padded
