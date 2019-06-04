import torch
import torch.nn as nn
import numpy as np
from preprocess import transform

#  from encoder import Encoder
#  from decoder import Decoder
from models.nmt import NMT
from eng2freDataset import Dataset

N_EPOCH = 50
BATCH_SIZE = 64
EMBEDDING_DIM = 300
HIDDEN_DIM = 1000
LR = 0.001
RESCHEDULED = False

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _main():
    dataset = Dataset("./data/eng-fra.txt", transform=transform)
    data_loader = dataset.get_loader(BATCH_SIZE, True)
    model = nn.DataParallel(
        NMT(
            dataset.get_source_vocab_size(),
            dataset.get_target_vocab_size(),
            EMBEDDING_DIM,
            HIDDEN_DIM,
            3,
            True,
            0.5,
            "lstm",
        ).to(DEVICE)
    )
    model.module.init_weight()
    criterion = nn.NLLLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(N_EPOCH):
        losses = []
        for i, (train_X, train_y, len_X, len_y) in enumerate(data_loader):
            train_X = train_X.to(DEVICE)
            train_y = train_y.to(DEVICE)

            hidden = model.module.init_hidden(train_X.shape[0])
            for h in hidden:
                h.to(DEVICE)
            preds = model(train_X, train_y, torch.tensor(len_X), hidden)

            loss = criterion(preds, train_y[:, 1:].reshape(-1))
            losses.append(loss.item())

            optimizer.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0
            )  # gradient clipping
            optimizer.step()

            if i % 200 == 0:
                print(
                    "[%02d/%d] [%03d/%d] mean_loss : %0.2f"
                    % (
                        epoch,
                        N_EPOCH,
                        i,
                        len(dataset) // BATCH_SIZE,
                        np.mean(losses),
                    )
                )
                losses = []


if __name__ == "__main__":
    _main()
