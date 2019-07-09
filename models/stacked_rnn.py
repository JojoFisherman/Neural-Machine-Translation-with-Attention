""" Implementation of ONMT RNN for Input Feeding Decoding """
import torch
import torch.nn as nn
from typing import Tuple


class StackedLSTM(nn.Module):
    """
    Our own implementation of stacked LSTM.
    Needed for the decoder, because we do input feeding from Luong 2015.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout_p: float = 0.2,
    ):
        """
        Args:
            input_dim (int): The input dimension (embedding + hidden).
            hidden_dim (int): The dimension of the hidden state.
            n_layers (int): The number of rnns stacking together.
            dropout_p (float): The dropout probability.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList()

        for i in range(n_layers):
            self.layers.append(nn.LSTMCell(input_dim, hidden_dim))
            input_dim = hidden_dim

    def forward(
        self,
        input_feed: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
    ):
        """ Compute one forward step of the LSTM decoder, without
            attention.

        Args:
            input_feed (torch.Tensor): The concatenated input of the
                attentional vectors at previous time step and
                inputs at the current time step
                ```(batch, hidden+embedding)```.
            hidden (Tuple[torch.Tensor, torch.Tensor]): The hidden state
                from previous timestep
                ```(batch, n_layers, n_dir * hidden_dim)```.
        Returns:
            output (torch.Tensor): The hidden state of the last layer of
                the LSTM
                ```(batch, hidden_dim)```.
            hidden (Tuple[torch.Tensor, torch.Tensor]): The hidden state of all
                layers of the LSTM
                ```(batch, n_layers, hidden_dim)```.
        """
        batch = input_feed.shape[0]
        h_0, c_0 = [h.transpose(0, 1) for h in hidden]
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input_feed, (h_0[i], c_0[i]))
            input_feed = h_1_i
            if i + 1 != self.n_layers:
                input_feed = self.dropout(input_feed)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1, 1)
        c_1 = torch.stack(c_1, 1)

        assert h_1.shape == (
            batch,
            self.n_layers,
            self.hidden_dim,
        ), f"hidden shape {h_1.shape} doesn't match"
        assert c_1.shape == h_1.shape, f"cell shape {c_1.shape} doesn't match"
        assert input_feed.shape == (
            batch,
            self.hidden_dim,
        ), f"output shape {input_feed.shape} doesn't match"
        assert torch.all(
            input_feed.eq(h_1[:, -1, :])
        ), "output doesn't match with the hidden state of the last layer"
        return input_feed, (h_1, c_1)

    def init_weight(self):
        for layer in self.layers:
            ih = (
                param.data
                for name, param in layer.named_parameters()
                if "weight_ih" in name
            )
            hh = (
                param.data
                for name, param in layer.named_parameters()
                if "weight_hh" in name
            )
            for t in ih:
                nn.init.xavier_uniform_(t)
            for t in hh:
                nn.init.orthogonal_(t)


class StackedGRU(nn.Module):
    """
    Our own implementation of stacked GRU.
    Needed for the decoder, because we do input feeding.
    """

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        """
        Args:
            n_layers (int): The number of rnns stacking together.
            input_dim (int): The input dimension (embedding + hidden).
            hidden_dim (int): The dimension of the hidden state.
            dropout_p (float): The dropout probability.
        """
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input_feed, hidden):
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input_feed, hidden[0][i])
            input_feed = h_1_i
            if i + 1 != self.num_layers:
                input_feed = self.dropout(input_feed)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)
        return input_feed, (h_1,)
