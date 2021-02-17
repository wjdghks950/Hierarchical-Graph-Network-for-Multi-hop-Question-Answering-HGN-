import random
import logging
import os

import torch
import numpy as np
from collections import Counter

from transformers import (
    BertConfig,
    BertModel,
    ElectraConfig,
    BertTokenizer,
    ElectraTokenizer,
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
    AlbertConfig,
    AlbertModel,
    AlbertTokenizer
)

MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),  # (BertConfig, BertForSequenceClassification, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'albert': (AlbertConfig, AlbertModel, AlbertTokenizer),
}

MODEL_PATH_MAP = {
    'bert': 'bert-base-uncased',  # 'bert-base-multilingual-cased'
}


def get_label(args):
    return [0, 1]


def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_score(preds, labels)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_score(preds, labels):
    return {
        "acc": simple_accuracy(preds, labels),
    }


def f1_score(pred, label):
    tp = Counter(pred) & Counter(label)
    num_tp = sum(tp.values())
    if num_tp == 0:
        return 0
    prec = float(num_tp) / len(pred)
    rec = float(num_tp) / len(label)
    f1 = (2 * prec * rec) / (prec + rec)
    return f1, prec, rec


def exact_match(pred, label):
    return pred == label


class EarlyStopping():
    def __init__(self, patience=0, verbose=0):
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose

    def validate(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print("Training process is stopped early...")
                return True
        else:
            self._step = 0
            self._loss = loss

        return False
