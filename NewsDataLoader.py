import torchtext
import os
import spacy
import random
import dill
import torch
import json
import torchtext.vocab as vocab
import torchtext.data as data
from torchtext.data import BucketIterator

spacy = spacy.load("en_core_web_sm")
SEED = 1024


def spacy_tokenize(x):
    return [tok.text for tok in spacy.tokenizer(x)]


TEXT = torchtext.data.Field(
    tokenize=spacy_tokenize, batch_first=True, include_lengths=True
)


class NewsDataset(data.Dataset):
    def __init__(
        self, path, exts, fields, max_src_len=100, debug=False, **kwargs
    ):
        fields = [("src", fields[0]), ("tgt", fields[1])]
        examples = []
        with open(path + exts[0], encoding="utf-8") as src_file, open(
            path + exts[1], encoding="utf-8"
        ) as tgt_file:
            for i, (src_line, tgt_line) in enumerate(zip(src_file, tgt_file)):
                src_line = src_line.strip()
                tgt_line = tgt_line.strip()
                examples.append(
                    data.Example.fromlist([src_line, tgt_line], fields)
                )
                if debug and i == 1000:
                    break

        super().__init__(examples, fields, **kwargs)
        self.examples = [
            example for example in self.examples if len(example.src) > 20
        ]
        for example in self.examples:
            example.src = example.src[:max_src_len]
            example.tgt = ["<sos>"] + example.tgt + ["<eos>"]


class NewsDataLoader:
    def __init__(
        self,
        datapath="data",
        use_save=False,
        embed_path=None,
        train=True,
        build_vocab=True,
        batch_size=64,
        val_size=0.2,
        max_src_len=100,
        debug=False,
    ):
        random.seed(SEED)
        state = random.getstate()

        self.fields = (TEXT, TEXT)
        save = True
        if use_save:
            save = False
            build_vocab = False
            with open("data/dataset.pickle", "rb") as f:
                text = dill.load(f)
                self.fields = (text, text)

        train_dataset = NewsDataset(
            path=os.path.join(datapath, "news_train"),
            exts=(".en", ".de"),
            fields=self.fields,
            max_src_len=max_src_len,
            debug=debug,
        )

        test_dataset = NewsDataset(
            path=os.path.join(datapath, "news_test"),
            exts=(".en", ".de"),
            max_src_len=max_src_len,
            fields=self.fields,
        )

        train_dataset, val_dataset = train_dataset.split(
            split_ratio=(1 - val_size), random_state=state
        )

        if build_vocab:
            # load custom word vectors
            if embed_path:
                path, embed = os.path.split(embed_path)
                vec = vocab.Vectors(embed, cache=path)
                TEXT.build_vocab(train_dataset, vectors=vec)
            else:
                TEXT.build_vocab(
                    train_dataset, vectors="glove.6B.300d", max_size=40000
                )

                ## DEBUG
                #  TEXT.build_vocab(train_dataset, max_size=40000)

        temp = BucketIterator.splits(
            (train_dataset, val_dataset),
            batch_size=batch_size,
            device=-1,
            sort_key=lambda x: len(x.src),
            sort_within_batch=True,
            repeat=False,
            shuffle=True,
        )

        self.test_dataloader = BucketIterator(
            test_dataset,
            batch_size=batch_size,
            device=-1,
            sort_key=lambda x: len(x.src),
            sort_within_batch=True,
            repeat=False,
            shuffle=False,
        )

        self.train_dataloader, self.val_dataloader = temp
        self.stoi = self.fields[0].vocab.stoi
        self.itos = self.fields[0].vocab.itos
        self.sos_id = self.stoi["<sos>"]
        self.eos_id = self.stoi["<eos>"]
        self.pad_id = self.stoi["<pad>"]

        if save:
            with open("data/dataset.pickle", "wb") as f:
                temp = {
                    "train": train_dataset,
                    "val": val_dataset,
                    "test": test_dataset,
                    "field": self.fields[0],
                }
                temp = self.fields[0]
                dill.dump(temp, f)

    def __iter__(self):
        if self.mode == "train":
            dataloader = self.train_dataloader
        elif self.mode == "val":
            dataloader = self.val_dataloader
        else:
            dataloader = self.test_dataloader

        for batch in dataloader:
            x = batch.src
            y = batch.tgt
            yield (x[0], x[1], y[0], y[1])

    @property
    def n_examples(self):
        n = 0
        if self.mode == "train":
            n = len(self.train_dataloader.dataset)
        elif self.mode == "val":
            n = len(self.val_dataloader.dataset)
        else:
            n = len(self.test_dataloader.dataset)
        return n

    def __len__(self):
        if self.mode == "train":
            dataloader = self.train_dataloader
        elif self.mode == "val":
            dataloader = self.val_dataloader
        else:
            dataloader = self.test_dataloader
        return len(dataloader)

    def __call__(self, mode):
        self.mode = mode.lower()
        return self


if __name__ == "__main__":
    dl = NewsDataLoader(debug=True, use_save=True)
    for x, len_x, y, len_y in dl("test"):
        import pdb

        pdb.set_trace()
