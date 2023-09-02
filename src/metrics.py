from typing import List, Union
import logging
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np


def compute_acc(preds: List[Union[int, str]], golds: List[Union[int, str]], prefix=None) -> dict:
    """
    Computes accuracy for predictions of classification tasks.
    Args:
        preds (List[Union[int, str]]):
            List of predictions, elements can be either integer or string.
        golds (List[Union[int, str]]):
             List of golds, elements can be either integer or string.
        prefix (str, optional):
            Metric name prefix of results.
    Returns:
        results (dict):
            A dict mapping from metric names from metric scores,
            metrics include accuracy (acc).
    """
    if not isinstance(preds[0], type(golds[0])):
        logging.warning(f"The element types of golds ({type(golds[0])}) and predictions ({type(preds[0])}) "
                        f"is different, this will cause invalid evaluation results.")
    accuracy = accuracy_score(y_true=golds, y_pred=preds)
    return {
        f"{prefix}/acc" if prefix else "acc": accuracy * 100,
    }


def compute_p_r_f1(preds: List[Union[int, str]], golds: List[Union[int, str]], prefix=None, pos_label=1) -> dict:
    """
    Computes precision, recall and f1 scores for predictions of classification tasks.

    Args:
        preds (List[Union[int, str]]):
            List of predictions, elements can be either integer or string.

        golds (List[Union[int, str]]):
             List of golds, elements can be either integer or string.

        prefix (str, optional):
            Metric name prefix of results.

        pos_label (optional, defaults to 1):
            The positive label, defaults to 1.

    Returns:
        results (dict):
            A dict mapping from metric names from metric scores in percentage,
            metrics include precision (p), recall (r) and f1.
    """
    if not isinstance(preds[0], type(golds[0])):
        logging.warning(f"The element types of golds ({type(golds[0])}) and predictions ({type(preds[0])}) "
                        f"is different, this will cause invalid evaluation results.")
    f1 = f1_score(y_true=golds, y_pred=preds, pos_label=pos_label)
    p = precision_score(y_true=golds, y_pred=preds, pos_label=pos_label)
    r = recall_score(y_true=golds, y_pred=preds, pos_label=pos_label)
    return {
        f"{prefix}/precision" if prefix else "precision": p * 100,
        f"{prefix}/recall" if prefix else "recall": r * 100,
        f"{prefix}/f1" if prefix else "f1": f1 * 100
    }


def speed_up_for_device_mapping(platform, preds, runtime_cpus, runtime_gpus, prefix=None):
    if platform == "amd":
        runtime_baselines = runtime_cpus
    else:
        runtime_baselines = runtime_gpus

    runtime_preds = [runtime_gpus[idx] if preds[idx] == 1 else runtime_cpus[idx] for idx, pred in enumerate(preds)]
    speed_ups = np.array(runtime_baselines) / np.array(runtime_preds)
    return {f"{prefix}/speed_up" if prefix else "speed_up": np.mean(speed_ups)}


def speed_up_for_thread_coarsening(preds, runtimes: List[dict], prefix=None):
    preds_cf = [2 ** pred for pred in preds]
    preds_runtimes = []
    for pred_cf, runtime in zip(preds_cf, runtimes):
        if pred_cf in runtime:
            preds_runtimes.append(runtime[pred_cf])
        else:
            preds_runtimes.append(runtime[1])
    # preds_runtimes = [runtime.get(pred_cf, default=runtime["1"]) for pred_cf, runtime in zip(preds_cf, runtimes)]
    baseline_runtimes = [runtime[1] for runtime in runtimes]
    speed_ups = np.array(baseline_runtimes) / np.array(preds_runtimes)
    return {f"{prefix}/speed_up" if prefix else "speed_up": np.mean(speed_ups)}


def compute_map(vectors, labels, prefix=None):
    """
    Computes Mean Average Precision.

    Args:
        vectors:
            Representation vectors of all examples, in [num_examples, hidden_size].
        labels:
            Labels, in [num_examples].
        prefix:
            Prefix of metric name.

    Returns:
        results (dict):
            A dict mapping from metric name to score.

    """
    assert len(vectors) == len(labels)
    num_examples = len(labels)

    # scores: [B, B]
    # computes dot-product of each vector with every other vectors
    scores = np.matmul(vectors, vectors.T)
    # the diagonal elements represent the similarity of a vector with itself,
    # which is set to a very low value (-1000000) to avoid considering it
    # when calculating the average precision later
    np.fill_diagonal(scores, -1000000)
    # stores the count of occurrences of each label in the labels array
    # which means that for each label, how many examples are in the all examples
    # that belongs to this label
    label_to_count = {}
    # iter over each vector
    for i in range(num_examples):
        if int(labels[i]) not in label_to_count:
            label_to_count[int(labels[i])] = 0
        else:
            label_to_count[int(labels[i])] += 1

    # sort scores by similarity
    # [B, B]
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]
    all_ap = []
    # iter over each example
    for i in range(num_examples):
        # label is the gold label of the current example
        label = int(labels[i])
        all_p_at_k = []
        # for j in range(label_to_count[label]):
        for j in range(num_examples - 1):
            # the j-th highest example for i-th example
            index = sort_ids[i, j]
            # if the j-th highest example is as the same label as the current example
            if int(labels[index]) == label:
                # precision@j = the number correctly retrieved / the number of examples with the same label
                p_at_k = (len(all_p_at_k) + 1) / (j + 1)
                all_p_at_k.append(p_at_k)
        ap = sum(all_p_at_k) / label_to_count[label]
        all_ap.append(ap)

    return {f"{prefix}/map" if prefix else "map": float(np.mean(all_ap)) * 100}
