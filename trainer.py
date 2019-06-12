import torch
import torch.nn as nn
import numpy as np
import argparse
from preprocess import transform
from utils import progress_bar, idx2sentence, save_checkpoint, get_params_dict
from metrics.metrics import evaluate_ppl, evaluate_bleu

#  from encoder import Encoder
#  from decoder import Decoder
from models.nmt import NMT
from eng2freDataset import Dataset
from NewsDataLoader import NewsDataLoader

N_EPOCH = 50
BATCH_SIZE = 64
EMBEDDING_DIM = 300
HIDDEN_DIM = 500
LR = 0.001
RESCHEDULED = False

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser()
parser.add_argument(
    "--resume", "-r", type=str, help="path of checkpoint to resume training"
)
parser.add_argument("--config", "-c", type=str, help="path of the config json")


def train(
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
    dataloader,
    metric=None,
    device=DEVICE,
    savename="model",
    pretrained_emb=None,
    resume_path=None,
    **kwargs,
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
        ).to(device)
    )

    start_epoch = 0
    model.module.init_weight()
    criterion = nn.NLLLoss(ignore_index=dataloader.pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5
    )
    if resume_path:
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        save_checkpoint.best = checkpoint["bleu"]
        start_epoch = checkpoint["epoch"] + 1
        for p in model.parameters():
            p.requires_grad = True

    for epoch in range(start_epoch, n_epochs):
        running_loss = 0.0
        running_total = 0
        n_predict_words = 0
        model.train()
        for i, (train_X, len_X, train_y, len_y) in enumerate(
            dataloader("train")
        ):
            train_X = train_X.to(DEVICE)
            train_y = train_y.to(DEVICE)
            hidden = model.module.init_hidden(train_X.shape[0])
            for h in hidden:
                h.to(device)

            log_p = model(
                train_X, train_y, len_X, hidden, teacher_forcing_ratio
            )
            # remove <sos>
            loss = criterion(log_p, train_y[:, 1:].reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0
            )  # gradient clipping
            optimizer.step()

            # statistics
            running_total += train_X.shape[0]
            running_loss += loss.item()
            n_predict_words += len_y.sum().item() - len_y.shape[0]
            _loss = running_loss / running_total

            progress_bar(
                running_total,
                dataloader.n_examples,
                epoch,
                n_epochs,
                {
                    "loss": _loss,
                    "ppl": evaluate_ppl(running_loss, n_predict_words),
                },
            )
        _score = validate(dataloader, model, criterion, epoch, device)
        save_checkpoint(
            {
                "net": model.state_dict(),
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "stoi": dataloader.stoi,
            },
            "bleu",
            _score,
            epoch,
            savename,
        )
        scheduler.step(_loss)


def validate(dataloader, model, loss_fn, epoch, device):
    model.eval()
    running_loss = 0.0
    running_total = 0
    running_scores = 0.0

    with torch.no_grad():
        for i, (val_X, len_X, val_y, len_y) in enumerate(dataloader("val")):
            val_X = val_X.to(device)
            val_y = val_y.to(device)

            hidden = model.module.init_hidden(val_X.shape[0])
            for h in hidden:
                h.to(device)

            # log_p: (batch , (tgt_len-1), tgt_vocab_size)
            log_p = model(val_X, val_y, len_X, hidden, 0.0)
            loss = loss_fn(log_p, val_y[:, 1:].reshape(-1))
            running_loss += loss.item()
            running_total += val_X.shape[0]

            log_p = log_p.reshape(val_y.shape[0], val_y.shape[1] - 1, -1)
            # decoded: (batch, tgt_len-1) without <sos>
            decoded = log_p.max(dim=2)[1]
            for true, pred in zip(val_y[:, 1:], decoded):
                running_scores += evaluate_bleu(
                    idx2sentence(true, dataloader.itos),
                    idx2sentence(pred, dataloader.itos),
                )

        _loss = running_loss / running_total
        _score = running_scores / running_total

        progress_bar(msg={"val-loss": _loss, "bleu": _score}, train=False)
    return _score


def _main():
    args = parser.parse_args()
    data_loader = NewsDataLoader(save_wordidx=True)

    if args.config:
        config = get_params_dict(args.config)

        train(
            dataloader=data_loader,
            **config,
            src_vocab_size=len(data_loader.stoi),
            tgt_vocab_size=len(data_loader.stoi),
            resume_path=args.resume,
        )

    #  train(
    #  data_loader,
    #  BATCH_SIZE,
    #  N_EPOCH,
    #  "lstm",
    #  True,
    #  3,
    #  HIDDEN_DIM,
    #  EMBEDDING_DIM,
    #  0.5,
    #  len(data_loader.stoi),
    #  len(data_loader.stoi),
    #  LR,
    #  0.5,
    #  None,
    #  DEVICE,
    #  resume_path=args.resume,
    #  )


if __name__ == "__main__":
    _main()
