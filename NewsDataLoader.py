import torchtext
import torchtext.data as data
import torchtext.vocab as vocab
import os
import spacy
import pandas as pd
import random
import dill
from tqdm import tqdm
from torchtext.data import BucketIterator

spacy = spacy.load("en_core_web_sm")
SEED = 1024


def spacy_tokenize(x):
    return [
        tok.text
        for tok in spacy.tokenizer(x)
        if not tok.is_punct | tok.is_space
    ]


class NewsDataset(data.Dataset):
    def __init__(
        self, path, max_src_len=100, field=None, debug=False, **kwargs
    ):
        examples = []
        fields = [("src", field), ("tgt", field)]
        df = pd.read_csv(path, encoding="utf-8", usecols=["content", "title"])
        df = df[~(df["content"].isnull() | df["title"].isnull())]
        df = df[~(df["content"] == "[]")]
        for i in tqdm(range(df.shape[0])):
            examples.append(
                data.Example.fromlist(
                    [df.iloc[i].content, df.iloc[i].title], fields
                )
            )
            if debug and i == 100:
                break
        super().__init__(
            examples, fields, filter_pred=lambda s: len(s.src) > 10, **kwargs
        )
        for example in self.examples:
            example.tgt = ["<sos>"] + example.tgt + ["<eos>"]


class NewsDataLoader:
    def __init__(
        self,
        csv_path,
        use_save=False,
        embed_path=None,
        build_vocab=True,
        batch_size=64,
        val_size=0.2,
        max_src_len=100,
        save=True,
        shuffle=True,
        debug=False,
    ):
        random.seed(SEED)

        def trim_sentence(s):
            return s[:max_src_len]

        self.field = data.Field(
            tokenize=spacy_tokenize,
            batch_first=True,
            include_lengths=True,
            lower=True,
            preprocessing=trim_sentence,
        )

        if use_save:
            save = False
            build_vocab = False
            with open("data/dataset.pickle", "rb") as f:
                self.field = dill.load(f)

        dataset = NewsDataset(
            csv_path, max_src_len, field=self.field, debug=debug
        )

        if build_vocab:
            # load custom word vectors
            if embed_path:
                path, embed = os.path.split(embed_path)
                vec = vocab.Vectors(embed, cache=path)
                self.field.build_vocab(dataset, vectors=vec)
            else:
                self.field.build_vocab(
                    dataset, vectors="glove.6B.300d", max_size=40000
                )

        self.dataloader = BucketIterator(
            dataset,
            batch_size=batch_size,
            device=-1,
            sort_key=lambda x: len(x.src),
            sort_within_batch=True,
            repeat=False,
            shuffle=shuffle,
        )
        self.stoi = self.field.vocab.stoi
        self.itos = self.field.vocab.itos
        self.sos_id = self.stoi["<sos>"]
        self.eos_id = self.stoi["<eos>"]
        self.pad_id = self.stoi["<pad>"]
        self.n_examples = len(dataset)

        if save:
            with open("data/dataset.pickle", "wb") as f:
                temp = self.field
                dill.dump(temp, f)

    def __iter__(self):
        for batch in self.dataloader:
            x = batch.src
            y = batch.tgt
            yield (x[0], x[1], y[0], y[1])

    def __len__(self):
        return len(self.dataloader)


if __name__ == "__main__":
    dl = NewsDataLoader(csv_path="data/val.csv", debug=True, save=False)
    for x, len_x, y, len_y in dl:
        if (len_x == 0).any():
            import pdb

            pdb.set_trace()
