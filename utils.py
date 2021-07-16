import logging
import os
import random

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    T5Config,
    T5Tokenizer,
    T5ForConditionalGeneration
)

MODEL_CLASSES = {
    "t5": (T5Config, T5ForConditionalGeneration, T5Tokenizer),
}

MODEL_PATH_MAP = {
    "t5": "t5-base",
}

def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)


def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(actuals, predictions):
    # assert len(intent_preds) == len(intent_labels) == len(slot_preds) == len(slot_labels)
    results = {}
    recalls = []
    precisions = []
    f1s = []
    for i, actual in enumerate(actuals):
        actual = actual.split('_')
        pred = predictions[i].split('-')
        match = len(set(actual) & set(pred))
        recall = match/len(actual)
        precision = match/len(pred)
        f1 = 2*recall*precision/(recall+precision)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)

    mac_recall = sum(recalls)/len(recalls)
    mac_precision = sum(precisions)/len(precisions)
    mac_f1 = sum(f1s)/len(f1s)
    results['recall'] = mac_recall
    results['precision'] = mac_precision
    results['f1'] = mac_f1
    return results

