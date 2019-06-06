import torch.nn as nn
import torch
from .encoder import Encoder
from .decoder import Decoder


class NMT(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        n_layers: int = 1,
        bidir: bool = False,
        dropout_p: float = 0.5,
        rnn_type: str = "lstm",
        **kwargs,
    ):
        """
        Args:
            src_vocab_size (int): The number of vocabularies in source side.
            tgt_vocab_size (int): The number of vocabularies in target side.
            embedding_dim (int): The dimension of the embeddings.
            hidden_dim (int): The dimension of the hidden state.
            n_layers (int): The number of rnns stacking together.
            bidir (bool): Use bi-directional.
            dropout_p (float): The dropout probability.
            rnn_type (str): The type of rnn to use (gru/lstm).
        """
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidir = bidir
        self.dropout_p = dropout_p
        self.rnn_type = rnn_type.lower()

        self.encoder = Encoder(
            src_vocab_size, embedding_dim, hidden_dim, n_layers, bidir=bidir
        )
        self.decoder = Decoder(
            tgt_vocab_size,
            embedding_dim,
            hidden_dim * 2,
            n_layers,
            dropout_p,
            rnn_type,
        )

    def init_hidden(self, batch_size: int):
        """ Initial hidden states for the encoders

        Returns:
            hidden: (Tuple[torch.Tensor, torch.Tensor])
                ```(batch, n_layers * n_dir, hidden_dim)```
        """
        return self.encoder.init_hidden(batch_size)

    def init_weight(self):
        self.encoder.init_weight()
        self.decoder.init_weight()

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        encoder_lengths: torch.Tensor,
        encoder_hidden: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ):
        """
        Args:
            x (torch.Tensor): A (batch, src_len) tensor containing
                the indexes of words in each training example
            y (torch.Tensor): A (batch, tgt_len) tensor, within which
                each sequence is a list of token IDs
            encoder_lengths (list): A (batch) list containing the sequence
                length of each training example
            encoder_hidden (torch.Tensor): A (batch, num_layers*num_directions,
                hidden_size) tensor of h_t-1
        Returns:
            log_p (torch.Tensor): The log probability in every time step
                ```(batch * (tgt_len-1), tgt_vocab_size)```


        """
        encoder_outputs, decoder_hidden = self.encoder(
            x, encoder_lengths, encoder_hidden
        )
        preds = self.decoder(
            y,
            decoder_hidden,
            encoder_outputs,
            encoder_lengths,
            teacher_forcing_ratio,
        )
        return preds
