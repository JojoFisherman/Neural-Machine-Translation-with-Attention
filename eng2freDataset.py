import torch
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence
from preprocess import transform


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, min_length=3, max_length=25, transform=None):
        corpus = [l.strip() for l in open(path, "r", encoding="utf-8")]

        self.raw_X, self.raw_y = [], []
        for l in corpus:
            X, y = l.split("\t")
            if transform:
                X = transform(X)
                y = transform(y)
            X = X.split()
            y = ["<s>"] + y.split()
            if (
                len(X) >= min_length
                and len(X) <= max_length
                and len(y) >= min_length
                and len(y) <= max_length
            ):
                self.raw_X.append(X)
                self.raw_y.append(y)

        source_vocab = list(
            set([word for sent in self.raw_X for word in sent])
        )
        target_vocab = list(
            set([word for sent in self.raw_y for word in sent])
        )

        self.source2index = {"<PAD>": 0, "<UNK>": 1, "<s>": 2, "</s>": 3}
        for v in source_vocab:
            if v not in self.source2index:
                self.source2index[v] = len(self.source2index)

        self.target2index = {"<PAD>": 0, "<UNK>": 1, "<s>": 2, "</s>": 3}
        for v in target_vocab:
            if v not in self.target2index:
                self.target2index[v] = len(self.target2index)

        self.index2target = {}
        for k, v in self.target2index.items():
            self.index2target[v] = k

    def __getitem__(self, index):
        train_X = [
            self.source2index[w]
            if w in self.source2index
            else self.source2index["<UNK>"]
            for w in self.raw_X[index]
        ]
        train_y = [
            self.target2index[w]
            if w in self.target2index
            else self.target2index["<UNK>"]
            for w in self.raw_y[index]
        ]
        return (
            torch.tensor(train_X).reshape(1, -1),
            torch.tensor(train_y).reshape(1, -1),
        )

    def __len__(self):
        return len(self.raw_X)

    def collate_fn(self, data):
        data.sort(key=lambda x: x[0].shape[1], reverse=True)
        train_X, train_y = list(zip(*data))
        train_X = [x.reshape(-1) for x in train_X]
        train_y = [y.reshape(-1) for y in train_y]
        len_X = [x.shape[0] for x in train_X]
        len_y = [y.shape[0] for y in train_y]
        train_X = pad_sequence(
            train_X, batch_first=True, padding_value=self.source2index["<PAD>"]
        )
        train_y = pad_sequence(
            train_y, batch_first=True, padding_value=self.target2index["<PAD>"]
        )
        return train_X, train_y, len_X, len_y

    def get_loader(self, batch_size, shuffle):
        return torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
        )

    def get_source_vocab_size(self):
        return len(self.source2index)

    def get_target_vocab_size(self):
        return len(self.target2index)


if __name__ == "__main__":
    ds = Dataset("data/eng-fra.txt", transform=transform)
    for i, (train_X, train_y, len_X, len_y) in enumerate(
        ds.get_loader(4, True)
    ):
        print(train_X)
        print(len_X)
        break
