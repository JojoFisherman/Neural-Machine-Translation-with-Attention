import torch
import torch.nn as nn
import numpy as np
from preprocess import transform
from encoder import Encoder
from decoder import Decoder
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
    encoder = nn.DataParallel(
        Encoder(
            dataset.get_source_vocab_size(), EMBEDDING_DIM, HIDDEN_DIM, 3, True
        ).to(DEVICE)
    )
    decoder = nn.DataParallel(
        Decoder(
            dataset.get_target_vocab_size(), EMBEDDING_DIM, HIDDEN_DIM * 2, 3
        ).to(DEVICE)
    )
    encoder.module.init_weight()
    decoder.module.init_weight()
    criterion = nn.NLLLoss(ignore_index=0)
    enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=LR)
    dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=LR)

    for epoch in range(N_EPOCH):
        losses = []
        for i, (train_X, train_y, len_X, len_y) in enumerate(data_loader):
            train_X = train_X.to(DEVICE)
            train_y = train_y.to(DEVICE)
            #  start_decode = torch.tensor(
            #  [dataset.target2index["<s>"]] * train_X.shape[0]
            #  ).reshape(-1, 1).to(DEVICE)

            hidden = encoder.module.init_hidden(train_X.shape[0])
            for h in hidden:
                h.to(DEVICE)
            output, hidden = encoder(train_X, torch.tensor(len_X), hidden)
            preds = decoder(train_y, hidden, output, torch.tensor(len_X))

            loss = criterion(preds, train_y[:, 1:].reshape(-1))
            losses.append(loss.item())

            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                encoder.parameters(), 5.0
            )  # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                decoder.parameters(), 5.0
            )  # gradient clipping
            enc_optimizer.step()
            dec_optimizer.step()

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
