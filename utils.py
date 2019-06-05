import torch
import spacy
import os
import json
from time import time
from typing import Dict


def spacy_tokenize(x):
    return [tok.text for tok in spacy.tokenizer(x)]


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (
        torch.arange(0, max_len)
        .type_as(lengths)
        .repeat(batch_size, 1)
        .lt(lengths.unsqueeze(1))
    )


def sentence2idx(sentence, stoi):
    if isinstance(sentence, str):
        sentence = spacy_tokenize(sentence)
    for w in sentence:
        w = stoi[w]
    return torch.tensor(sentence)


def idx2sentence(idxes: torch.Tensor, itos) -> str:
    sentence = []
    for idx in idxes:
        sentence.append(itos[idx.item()])
    return " ".join(sentence)


def progress_bar(
    batch_size=0,
    value=1,
    endvalue=1,
    epoch=1,
    n_epoch=1,
    msg: Dict[str, float] = {},
    train=True,
    bar_length=20,
):
    if "start_time" not in progress_bar.__dict__ or value == 0:
        progress_bar.start_time = time()

    t = time() - progress_bar.start_time
    percent = value / endvalue
    arrow = "=" * int((percent * bar_length) - 1) + ">"
    spaces = " " * (bar_length - len(arrow))
    msg = " - ".join([f"{k}: {v:.4f}" for k, v in msg.items()])

    if value == batch_size:
        print(f"Epoch {epoch+1}/{n_epoch}", flush=True)
        progress_bar.start_time = time()

    if train:
        print(
            f"{value}/{endvalue} [{arrow + spaces}] - {(t):.4f}s - {msg}",
            end="" if value == endvalue else "\r",
            flush=True,
        )
    elif msg:
        print(f" - {msg}", flush=True)


def save_checkpoint(state, metric, value, epoch, filename="checkpoint.pth"):
    if "best" not in save_checkpoint.__dict__:
        save_checkpoint.best = 0.0
    if not os.path.isdir("save"):
        os.mkdir("save")
    state[metric] = save_checkpoint.best

    if value > save_checkpoint.best:
        print("Saving best model...")
        save_checkpoint.best = value
        state[metric] = value
        torch.save(
            state,
            #  os.path.join('checkpoint', f'model_best_{metric}_{value}.pth')
            os.path.join("save", f"{filename}_best.pth"),
        )

    torch.save(state, os.path.join("save", f"{filename}_checkpoint.pth"))


def get_params_dict(filename):
    j = {}
    if os.path.isfile(filename):
        with open(filename, "r") as f:
            j = json.load(f)
    return j
