from models import nmt
from utils import get_params_dict
import torch
import json

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(path, config_file: str, vocab_file: str):
    with open(vocab_file, "r", encoding="utf-8") as f:
        stoi = json.load(f)
    config = get_params_dict(config_file)
    model = nmt(**config, src_vocab_size=len(stoi), tgt_vocab_size=len(stoi))
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["net"])
