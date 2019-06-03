import torch
import torch.nn as nn
from typing import List
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
    pad_sequence,
)


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        n_layers: int = 1,
        bidir: bool = False,
        rnn_type: str = "lstm",
    ):
        """
        Args:
            vocab_size (int): The number of vocabularies.
            embedding_dim (int): The dimension of the embeddings.
            hidden_dim (int): The dimension of the hidden state.
            n_layers (int): The number of rnns stacking together.
            bidir (bool): Use bi-directional.
            rnn_type (str): The type of rnn to use (gru/lstm).
        """
        super().__init__()

        self.n_dir = 2 if bidir else 1
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn_type = rnn_type.lower()
        if rnn_type.lower() == "gru":
            rnn = nn.GRU
        elif rnn_type.lower() == "lstm":
            rnn = nn.LSTM
        else:
            raise ValueError("rnn_type should be either gru or lstm")
        self.rnn = rnn(
            embedding_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            bidirectional=bidir,
        )

    def init_hidden(self, batch_size: int):
        # gru hidden shape: (batch, n_layers * n_directions, hidden_size)
        # lstm hidden shape: 2x(batch, n_layers * n_directions, hidden_size)
        if self.rnn_type.lower() == "gru":
            return torch.zeros(
                batch_size, self.n_layers * self.n_dir, self.hidden_dim
            )
        else:
            return tuple(
                [
                    torch.zeros(
                        batch_size, self.n_layers * self.n_dir, self.hidden_dim
                    )
                    for _ in range(2)
                ]
            )

    def init_weight(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        ih = (
            param.data
            for name, param in self.rnn.named_parameters()
            if "weight_ih" in name
        )
        hh = (
            param.data
            for name, param in self.rnn.named_parameters()
            if "weight_hh" in name
        )
        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)

    def forward(
        self, inputs: torch.Tensor, input_lengths: list, hidden: torch.Tensor
    ):
        """
        Args:
            inputs (torch.Tensor): A (batch, seq_len) tensor containing
                the indexes of words in each training example
            input_lengths (list): A (batch) list containing the sequence length
                of each training example
            hidden (torch.Tensor): A (batch, num_layers*num_directions,
                hidden_size) tensor of h_t-1

        Returns:
            outputs (torch.Tensor): A (batch, seq_len,
                n_directions * hidden_size) tensor containing the hidden states
                from the last layer of all time steps
            hidden (torch.Tensor): A (batch, n_layers,
                n_direction * hidden_size) tensor containing the hidden state
                of the last time step

        """
        if self.rnn_type == "gru":
            hidden = hidden.transpose(1, 0).contiguous()
        else:
            hidden = tuple([h.transpose(1, 0).contiguous() for h in hidden])

        total_length = inputs.shape[1]
        embedded = self.embedding(inputs)
        packed = pack_padded_sequence(
            embedded, input_lengths, batch_first=True
        )
        # outputs: (batch, seq_len, n_directions * hidden_size)
        # hidden: (n_layers * n_directions, batch, hidden_size)
        outputs, hidden = self.rnn(packed, hidden)
        outputs, _ = pad_packed_sequence(
            outputs, batch_first=True, total_length=total_length
        )

        # (n_layers * n_directions, batch, hidden_size) ->
        # (batch, n_layers, hidden_size * n_directions)
        def _cat_directions(hidden):
            hidden = torch.cat(
                [
                    hidden[0 : hidden.shape[0] : 2],
                    hidden[1 : hidden.shape[0] : 2],
                ],
                2,
            ).transpose(0, 1)
            return hidden

        if self.n_dir == 2:
            if self.rnn_type == "gru":
                hidden = _cat_directions(hidden)
            else:
                hidden = tuple([_cat_directions(h) for h in hidden])

        assert outputs.shape == (
            inputs.shape[0],
            total_length,
            self.hidden_dim * self.n_dir,
        ), f"outputs shape {outputs.shape} doesn't match"
        for h in hidden:
            assert h.shape == (
                inputs.shape[0],
                self.n_layers,
                self.hidden_dim * self.n_dir,
            )
        return outputs, hidden


# testing
if __name__ == "__main__":
    N_EPOCH = 50
    BATCH_SIZE = 64
    EMBEDDING_DIM = 4
    HIDDEN_DIM = 30
    LR = 0.001

    from eng2freDataset import Dataset
    from preprocess import transform

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ds = Dataset("./data/eng-fra.txt", transform=transform)
    data_loader = ds.get_loader(BATCH_SIZE, True)
    train_X, train_y, len_X, len_y = list(iter(data_loader))[0]
    encoder = nn.DataParallel(
        Encoder(
            ds.get_source_vocab_size(), EMBEDDING_DIM, HIDDEN_DIM, 3, True
        ).to(device),
        dim=0,
    )
    hidden = encoder.module.init_hidden(BATCH_SIZE)
    outputs, hidden = encoder(train_X, torch.tensor(len_X), hidden)
    print(hidden.shape)
    print(outputs.shape)
