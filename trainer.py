import torch
import torch.nn as nn
import numpy as np
from preprocess import transform
from utils import progress_bar
from metrics.metrics import evaluate_ppl

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


def train(
    train_dataloader,
    val_dataloader,
    batch_size,
    n_epochs,
    rnn_type,
    bidir,
    n_layers,
    hidden_dim,
    embedding_dim,
    teacher_forcing_ratio,
    src_vocab_size,
    tgt_vocab_size,
    learning_rate,
    dropout_p,
    metric,
    savename,
    device,
    pretrained_emb=None,
):
    model = nn.DataParallel(
        NMT(
            src_vocab_size,
            tgt_vocab_size,
            embedding_dim,
            hidden_dim,
            n_layers,
            bidir,
            dropout_p,
            rnn_type.lower(),
            teacher_forcing_ratio,
        ).to(device)
    )
    model.module.init_weight()
    criterion = nn.NLLLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(n_epochs):
        losses = []
        for i, (train_X, train_y, len_X, len_y) in enumerate(train_dataloader):
            train_X = train_X.to(DEVICE)
            train_y = train_y.to(DEVICE)
            hidden = model.module.init_hidden(train_X.shape[0])
            for h in hidden:
                h.to(device)

            log_p = model(train_X, train_y, torch.tensor(len_X), hidden)
            # remove <sos>
            loss = criterion(log_p, train_y[:, 1:].reshape(-1))
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0
            )  # gradient clipping
            optimizer.step()


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
