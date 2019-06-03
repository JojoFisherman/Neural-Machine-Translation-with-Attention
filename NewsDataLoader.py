import torchtext
import os
import spacy
import random
import dill
import torch
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
    def __init__(self, path, exts, fields, **kwargs):
        fields = [("src", fields[0]), ("tgt", fields[1])]
        examples = []
        with open(path + exts[0], encoding="utf-8") as src_file, open(
            path + exts[1], encoding="utf-8"
        ) as tgt_file:
            for src_line, tgt_line in zip(src_file, tgt_file):
                src_line = src_line.strip()
                tgt_line = tgt_line.strip()
                examples.append(
                    data.Example.fromlist([src_line, tgt_line], fields)
                )
        super().__init__(examples, fields, **kwargs)
        for example in examples:
            example.tgt = ["<sos>"] + example.tgt + ["<eos>"]


class NewsDataLoader:
    def __init__(
        self,
        fields,
        datapath="data",
        load_field_path=None,
        save_field_path=None,
        embed_path=None,
        train=True,
        build_vocab=True,
        batch_size=64,
        val_size=0.2,
    ):
        random.seed(SEED)
        state = random.getstate()

        if load_field_path:
            build_vocab = False
            with open(load_field_path, "rb") as f:
                field = dill.load(f)
                fields = (field, field)

        train_dataset = NewsDataset(
            path=os.path.join(datapath, "news_train"),
            exts=(".en", ".de"),
            fields=fields,
        )

        test_dataset = NewsDataset(
            path=os.path.join(datapath, "news_test"),
            exts=(".en", ".de"),
            fields=fields,
        )

        train_dataset, val_dataset = train_dataset.split(
            split_ratio=(1 - val_size), random_state=state
        )

        if build_vocab:
            # load custom word vectors
            if embed_path:
                path, embed = os.path.split(embed_path)
                vec = vocab.Vectors(embed, cache=path)
                self.field.build_vocab(train_dataset, vectors=vec)
            else:
                self.field.build_vocab(train_dataset, vectors="glove.6B.300d")

        if save_field_path:
            with open(save_field_path, "wb") as f:
                dill.dump(fields[0], f)

        temp = BucketIterator.splits(
            (train_dataset, val_dataset, test_dataset),
            batch_size=batch_size,
            device=-1,
            sort_key=lambda x: len(x.src),
            sort_within_batch=True,
            repeat=False,
            shuffle=True,
        )
        self.train_dataloader, self.val_dataloader, self.test_dataloader = temp
        self.stoi = fields[0].vocab.stoi
        self.itos = fields[0].vocab.itos
        self.sos_id = self.stoi["<sos>"]
        self.eos_id = self.stoi["<eos>"]


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
            yield (x, y)

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
    dl = NewsDataLoader((TEXT, TEXT), load_field_path="data/field.pkl")
    for x, y in dl("train"):
        import pdb

        pdb.set_trace()
