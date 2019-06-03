import torch
import spacy


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
