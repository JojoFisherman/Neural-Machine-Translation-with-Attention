from models.nmt import NMT
from utils import get_params_dict, idx2sentence, progress_bar
from translate import beam_search
from NewsDataLoader import NewsDataLoader
from metrics.metrics import evaluate_ppl, evaluate_bleu
import torch
import json
import argparse
import pdb


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(path, config_file: str):
    config = get_params_dict(config_file)
    checkpoint = torch.load(path)
    itos = checkpoint["itos"]
    model = NMT(**config, src_vocab_size=len(itos), tgt_vocab_size=len(itos))
    model.load_state_dict(checkpoint["net"])
    return model, itos


def test(model, dataloader, itos, device=DEVICE):
    model = model.to(device)
    stoi = {}
    for i, s in enumerate(itos):
        stoi[s] = i

    model.eval()
    running_total = 0
    running_scores = 0.0
    with torch.no_grad():
        for test_X, len_X, test_y, len_y in dataloader("test"):
            for x, _len_x, y, _len_y in zip(test_X, len_X, test_y, len_y):
                running_total += 1
                x = idx2sentence(x, itos)
                # remove <sos> and <eos>
                y = idx2sentence(y, itos)[1 : (_len_y - 1)]

                best_hypothesis = beam_search(
                    stoi, itos, model.encoder, model.decoder, x, 5, 20
                )[0]

                pdb.set_trace()

                running_scores += evaluate_bleu(y, best_hypothesis.value)
                _score = running_scores / running_total
                progress_bar(
                    running_total,
                    dataloader.n_examples,
                    0,
                    1,
                    msg={"bleu": _score},
                    train=True,
                )
    progress_bar(train=False)
    final_score = running_scores / running_total
    return final_score


if __name__ == "__main__":
    model, itos = load_model("save/model_best.pth", "config.json")
    dataloader = NewsDataLoader(use_save=True, debug=True)
    test(model, dataloader, itos)
