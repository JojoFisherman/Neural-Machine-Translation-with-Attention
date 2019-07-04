import torch
import spacy
import os
import json
from time import time
from typing import Dict

spacy = spacy.load("en_core_web_sm")


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


def get_params_dict(filename):
    j = {}
    if os.path.isfile(filename):
        with open(filename, "r") as f:
            j = json.load(f)
    return j


def load_model(path, config_file: str):
    from models.nmt import NMT

    config = get_params_dict(config_file)
    checkpoint = torch.load(path)
    itos = checkpoint["itos"]
    model = NMT(**config, src_vocab_size=len(itos), tgt_vocab_size=len(itos))
    model.load_state_dict(checkpoint["net"])
    return model, itos


def sentence2idx(sentence, stoi):
    if isinstance(sentence, str):
        sentence = spacy_tokenize(sentence)
    temp = [[]]
    for w in sentence:
        #  if w not in stoi:
        #  print(w + " not found in vocabulary")
        temp[0].append(stoi.get(w, stoi["<unk>"]))
    #  sentence = [[stoi.get(w, stoi["<unk>"]) for w in sentence]]
    return torch.tensor(temp)


def idx2sentence(idxes: torch.Tensor, itos) -> str:
    sentence = []
    for idx in idxes:
        sentence.append(itos[idx.item()])
    return sentence
    #  return " ".join(sentence)


def progress_bar(
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

    if "start" not in progress_bar.__dict__:
        progress_bar.start = True

    t = time() - progress_bar.start_time
    percent = value / endvalue
    arrow = "=" * int((percent * bar_length) - 1) + ">"
    spaces = " " * (bar_length - len(arrow))
    msg = " - ".join(
        [
            f"{k}: {v:.4f}" if k != "lr" else f"{k}: {v:.2E}"
            for k, v in msg.items()
        ]
    )

    #  if value == batch_size:
    if progress_bar.start:
        progress_bar.start = False
        print(f"Epoch {epoch+1}/{n_epoch}", flush=True)
        #  progress_bar.start_time = time()

    if train:
        print(
            f"{value}/{endvalue} [{arrow + spaces}] - {(t):.4f}s - {msg}",
            end="" if value == endvalue else "\r",
            flush=True,
        )
        return
    if msg:
        print(f" - {msg}", end="", flush=True)

    print()
    progress_bar.start = True
    progress_bar.start_time = time()


def save_checkpoint(state, metric, value, epoch, filename="model"):
    if "best" not in save_checkpoint.__dict__:
        save_checkpoint.best = 0.0
    if not os.path.isdir("save"):
        os.mkdir("save")
    state[metric] = save_checkpoint.best

    if value > save_checkpoint.best:
        print("Saving best model...")
        save_checkpoint.best = value
        state[metric] = value
        torch.save(state, os.path.join("save", f"{filename}_best.pth"))

    torch.save(state, os.path.join("save", f"{filename}_checkpoint.pth"))


def get_lr(optimizer):
    """ Get the current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]
