from utils import idx2sentence, progress_bar, load_model
from translate import beam_search
from NewsDataLoader import NewsDataLoader
from metrics.metrics import evaluate_ppl, evaluate_bleu
import torch
import json
import argparse


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test(model, dataloader, itos, device=DEVICE):
    model = model.to(device)
    stoi = {}
    for i, s in enumerate(itos):
        stoi[s] = i

    model.eval()
    running_total = 0
    running_scores = 0.0
    with torch.no_grad():
        #  for test_X, len_X, test_y, len_y in dataloader("test"):
        # debug
        for test_X, len_X, test_y, len_y in dataloader:
            for x, _len_x, y, _len_y in zip(test_X, len_X, test_y, len_y):
                running_total += 1
                x = idx2sentence(x, itos)
                # remove <sos> and <eos>
                y = idx2sentence(y, itos)[1 : (_len_y - 1)]

                best_hypothesis = beam_search(
                    stoi, itos, model.encoder, model.decoder, x, 5, 20
                )[0]

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
    model, itos = load_model("save/model_checkpoint.pth", "config.json")
    dataloader = NewsDataLoader(
        csv_path="data/train.csv", use_save=True, debug=True, build_vocab=False
    )
    #  dataloader = NewsDataLoader(use_save=True, debug=True)
    test(model, dataloader, itos)
