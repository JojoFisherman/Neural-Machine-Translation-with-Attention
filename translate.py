import torch
import torch.nn as nn
import pandas as pd
import argparse
from tqdm import tqdm
from NewsDataLoader import NewsDataLoader, spacy_tokenize
from typing import List, Dict
from utils import idx2sentence, sentence2idx, load_model
from collections import namedtuple

Hypothesis = namedtuple("Hypothesis", ["value", "score"])
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    help="Path for the model weights file",
    default="save/model_best.pth",
)
parser.add_argument(
    "--input",
    type=str,
    help="Path for the input csv, the header for "
    "the content of the articles should be named `content`",
    default="data/test.csv",
)

parser.add_argument(
    "--output",
    type=str,
    help="Path for the output csv",
    default="data/test_output.csv",
)

parser.add_argument(
    "--live",
    action="store_true",
    help="Feeding in the articles interactively "
    "and manually through the command line",
)


def beam_search(
    stoi: Dict[str, int],
    itos: List[str],
    encoder: nn.Module,
    decoder: nn.Module,
    input_sentence: List[str],
    n_beam: int,
    max_len: int,
):
    """
    Args:
        stoi: (Dict[str, int]): A dict that maps string to its index
        itos: (List[str]): A list that maps index to the string
        encoder (nn.Module): The encoder object.
        decoder (nn.Module): The decoder object.
        input_sentence (List[str]): A single sentence.
        n_beam (int): Number of beams.
        max_len (int): Maximum length for decoding.

    Returns:
        List[Hypothesis]: All the finished hypothesis("value", "score").
    """
    inputs = sentence2idx(input_sentence, stoi).to(DEVICE)
    hidden = [h.to(DEVICE) for h in encoder.init_hidden(1)]

    encoder_length = torch.tensor([inputs.shape[1]]).to(inputs.device)

    encoder_outputs, hidden = encoder(inputs, encoder_length, hidden)

    hypotheses = [["<sos>"]]
    hyp_scores = torch.zeros(len(hypotheses)).to(DEVICE)
    completed_hypotheses = []
    attn_h = torch.zeros(1, decoder.hidden_dim).to(DEVICE)

    t = 0
    while len(completed_hypotheses) < n_beam and t < max_len:
        t += 1
        n_hyp = len(hypotheses)

        # (n_hyp, src_len, 2h)
        exp_encoder_outputs = encoder_outputs.expand(
            n_hyp, encoder_outputs.shape[1], encoder_outputs.shape[2]
        )
        exp_encoder_length = encoder_length.expand(n_hyp)

        # (n_hyp)
        decoder_input = torch.tensor([stoi[hyp[-1]] for hyp in hypotheses]).to(
            DEVICE
        )
        emb_input = decoder.embedding(decoder_input)
        emb_input = decoder.dropout(emb_input)

        hidden, attn_h, log_p = decoder.step(
            attn_h, emb_input, hidden, exp_encoder_outputs, exp_encoder_length
        )

        live_hyp_num = n_beam - len(completed_hypotheses)
        contiuating_hyp_scores = (
            hyp_scores.unsqueeze(1).expand_as(log_p) + log_p
        ).view(-1)
        top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(
            contiuating_hyp_scores, k=live_hyp_num
        )

        prev_hyp_ids = top_cand_hyp_pos / len(stoi)
        hyp_word_ids = top_cand_hyp_pos % len(stoi)

        new_hypotheses = []
        live_hyp_ids = []
        new_hyp_scores = []

        for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(
            prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores
        ):
            prev_hyp_id = prev_hyp_id.item()
            hyp_word_id = hyp_word_id.item()
            cand_new_hyp_score = cand_new_hyp_score.item()

            hyp_word = itos[hyp_word_id]
            new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
            if hyp_word == "<eos>":
                completed_hypotheses.append(
                    Hypothesis(
                        value=new_hyp_sent[1:-1], score=cand_new_hyp_score
                    )
                )
            else:
                new_hypotheses.append(new_hyp_sent)
                live_hyp_ids.append(prev_hyp_id)
                new_hyp_scores.append(cand_new_hyp_score)

        if len(completed_hypotheses) == n_beam:
            break

        live_hyp_ids = torch.tensor(live_hyp_ids)

        hidden = tuple([h[live_hyp_ids] for h in hidden])
        attn_h = attn_h[live_hyp_ids]

        hypotheses = new_hypotheses
        hyp_scores = torch.tensor(new_hyp_scores).to(DEVICE)

    completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)
    if len(completed_hypotheses) == 0:
        completed_hypotheses.append(
            Hypothesis(value=hypotheses[0][1:], score=hyp_scores[0].item())
        )
    return completed_hypotheses


def _main():
    args = parser.parse_args()
    model, itos = load_model(args.model, "config.json")
    model = model.to(DEVICE)
    stoi = {}
    for i, s in enumerate(itos):
        stoi[s] = i

    if args.live:
        while True:
            sentence = input("Enter the content(type q to exit): ")
            if sentence == "q":
                break
            x = spacy_tokenize(sentence)
            best_hypothesis = beam_search(
                stoi, itos, model.encoder, model.decoder, x, 8, 20
            )[0]
            print("Predicted headline: ", best_hypothesis.value)
    else:
        df = pd.read_csv(args.input, encoding="utf-8")
        df = df[~(df["content"].isnull())]
        df = df[~(df["content"] == "[]")]
        contents = df["content"].str.lower().tolist()
        sentences = [spacy_tokenize(s)[:100] for s in contents]
        titles = []
        for i, x in tqdm(enumerate(sentences)):
            best_hypothesis = beam_search(
                stoi, itos, model.encoder, model.decoder, x, 8, 20
            )[0]
            titles.append(" ".join(best_hypothesis.value))
        df = pd.DataFrame({"content": contents, "title": titles})
        df.to_csv(args.output, encoding="utf-8", index=False)


if __name__ == "__main__":
    _main()
