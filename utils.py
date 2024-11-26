import os
import json

import random
import torch
import datasets

import numpy as np

from typing import Optional, Dict, List, Union, Tuple

def _get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size,
                sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens
                with a value of -100 are ignored.
                Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per 
                          (non-masked) token. Otherwise, return the sum of the 
                          log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log
        probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)

def get_batch_metrics(
    model,
    batch: Dict[str, Union[List, torch.LongTensor]],
    train=True
) -> Tuple[torch.FloatTensor, Dict[str, List]]:
    """Compute the SFT loss and other metrics for the given batch of inputs.
    """

    metrics = {}
    train_test = 'train' if train else 'eval'


    policy_chosen_logits = model(
        input_ids=batch['chosen_input_ids'],
        attention_mask=batch['chosen_attention_mask']
    ).logits.to(torch.float32)
    policy_chosen_logps = _get_batch_logps(
        policy_chosen_logits, batch['chosen_labels'],
        average_log_prob=False
    )
    losses = -policy_chosen_logps
    
    with torch.no_grad():
        for k in [
            'rejected', 'random', 'paraphrase', 'variant', 'nonresponse'
        ]:
            policy_predict_logtis = model(
                input_ids=batch[f'{k}_input_ids'],
                attention_mask=batch[f'{k}_attention_mask']
            ).logits.detach().to(torch.float32)
            policy_predict_logps = _get_batch_logps(
                policy_predict_logtis, batch[f'{k}_labels'],
                average_log_prob=False
            )
            del policy_predict_logtis
            metrics[f'logps_{train_test}/{k}'] = \
                policy_predict_logps.cpu().numpy().tolist()

    policy_chosen_logps = policy_chosen_logps.detach()

    metrics[f'logps_{train_test}/chosen'] = \
        policy_chosen_logps.cpu().numpy().tolist()

    all_devices_losses = losses.detach()

    metrics[f'loss/{train_test}'] = \
        all_devices_losses.cpu().numpy().tolist()

    return losses.mean(), metrics

class TemporarilySeededRandom:
    def __init__(self, seed):
        """Temporarily set the random seed, and then restore it when exiting the context."""
        self.seed = seed
        self.stored_state = None
        self.stored_np_state = None

    def __enter__(self):
        # Store the current random state
        self.stored_state = random.getstate()
        self.stored_np_state = np.random.get_state()

        # Set the random seed
        random.seed(self.seed)
        np.random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore the random state
        random.setstate(self.stored_state)
        np.random.set_state(self.stored_np_state)