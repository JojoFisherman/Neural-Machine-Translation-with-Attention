import torch
import torch.nn as nn
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import argparse
from torch.utils.tensorboard import SummaryWriter
from preprocess import transform
from utils import (
    progress_bar,
    idx2sentence,
    save_checkpoint,
    get_lr,
    get_params_dict,
)
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
WRITER = SummaryWriter(log_dir="./logs")
torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser()
parser.add_argument(
    "--resume", "-r", action="store_true", help="Resume training"
)
parser.add_argument("--debug", action="store_true", help="Debug mode")
parser.add_argument(
    "--savename",
    "-s",
    type=str,
    help="name for the model weights file",
    default="model",
)


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
    train_dataloader,
    val_dataloader,
    metric=None,
    device=DEVICE,
    savename="model",
    pretrained_emb=None,
    is_resume=False,
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
            pretrained_emb,
        ).to(device)
    )
    print("num of vocab:", src_vocab_size)

    start_epoch = 0
    model.module.init_weight()

    criterion = nn.NLLLoss(ignore_index=train_dataloader.pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5
    )
    if is_resume:
        checkpoint = torch.load(os.path.join("save", "model_checkpoint.pth"))
        model.module.load_state_dict(checkpoint["net"])
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
        for i, (train_X, len_X, train_y, len_y) in enumerate(train_dataloader):
            train_X = train_X.to(DEVICE)
            train_y = train_y.to(DEVICE)
            hidden = [
                h.to(device)
                for h in model.module.init_hidden(train_X.shape[0])
            ]

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
            _loss = running_loss / (i + 1)

            progress_bar(
                running_total,
                train_dataloader.n_examples,
                epoch + 1,
                n_epochs,
                {
                    "loss": _loss,
                    "lr": get_lr(optimizer),
                    "ppl": evaluate_ppl(running_loss, n_predict_words),
                },
            )

        WRITER.add_scalar("train_loss", _loss, epoch + 1)
        _score = validate(val_dataloader, model, criterion, epoch, device)
        scheduler.step(_loss)
        save_checkpoint(
            {
                "net": model.module.state_dict(),
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "itos": train_dataloader.itos,
            },
            "bleu",
            _score,
            epoch,
            savename,
        )


def validate(dataloader, model, loss_fn, epoch, device):
    model.eval()
    running_loss = 0.0
    running_total = 0
    running_scores = 0.0

    with torch.no_grad():
        for i, (val_X, len_X, val_y, len_y) in enumerate(dataloader):
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

    _loss = running_loss / i + 1
    _score = running_scores / running_total
    WRITER.add_scalar("val_loss", _loss, epoch + 1)
    WRITER.add_scalar("val_bleu", _score, epoch + 1)

    progress_bar(msg={"val-loss": _loss, "bleu": _score}, train=False)
    return _score


def _main():
    args = parser.parse_args()

    config = get_params_dict("config.json")
    if args.resume:
        train_dataloader = NewsDataLoader(
            csv_path="data/train.csv", use_save=True, debug=args.debug
        )
    else:
        train_dataloader = NewsDataLoader(
            csv_path="data/train.csv", save=True, debug=args.debug
        )

    val_dataloader = NewsDataLoader(
        csv_path="data/val.csv",
        build_vocab=False,
        use_save=True,
        debug=args.debug,
    )

    train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        **config,
        src_vocab_size=len(train_dataloader.stoi),
        tgt_vocab_size=len(train_dataloader.stoi),
        is_resume=args.resume,
        pretrained_emb=train_dataloader.field.vocab.vectors,
        savename=args.savename,
    )


if __name__ == "__main__":
    _main()
