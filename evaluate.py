import os
import json

import torch
import numpy
import wandb

from collections import defaultdict
from transformers import AutoModel, AutoTokenizer
from typing import Optional, Dict, List, Union, Tuple

from preference_datasets import get_batch_iterator
from utils import get_batch_metrics

def test():
    all_eval_metrics = defaultdict(list)
    # load the model and tokenizer
    model = AutoModel.from_pretrained('Qwen/Qwen2.5-7B').to('cuda').eval()
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B')

    data_iterator_kwargs = dict(
        names=['hh'],
        tokenizer=tokenizer,
        shuffle=True,
        max_length=2048,
        max_prompt_length=1024,
    )

    eval_iterator = get_batch_iterator(
        **data_iterator_kwargs,
        split='test',
        n_examples=256,
        batch_size=8,
        silent=False,
    )

    eval_batches = list(eval_iterator)

    with torch.no_grad():
        for eval_batch in eval_batches:
            # move the batch to the device
            eval_batch = {k: v.to('cuda') for k, v in eval_batch.items() if 'ids' in k}
            _, eval_metrics = get_batch_metrics(
                model, eval_batch, train=False
            )

            for k, v in eval_metrics.items():
                all_eval_metrics[k].extend(v)


if __name__ == '__main__':
    test()