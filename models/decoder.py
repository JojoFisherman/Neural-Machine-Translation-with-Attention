import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from .attention import Attention
from .stacked_rnn import StackedLSTM
from torch.nn.utils.rnn import pack_padded_sequence


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        embedding: nn.Module,
        hidden_dim: int,
        n_layers: int = 1,
        dropout_p: float = 0.2,
        rnn_type: str = "lstm",
    ):
        """
        Args:
            vocab_size (int): The number of vocabularies.
            embedding_dim (int): The dimension of the embeddings.
            hidden_dim (int): The dimension of the hidden state.
            n_layers (int): The number of rnns stacking together.
            dropout_p (float): The dropout probability.
            rnn_type (str): The type of rnn to use (gru/lstm).
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #  self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding = embedding
        self.dropout = nn.Dropout(dropout_p)
        if rnn_type.lower() == "gru":
            rnn = nn.GRU
        elif rnn_type.lower() == "lstm":
            rnn = StackedLSTM
        else:
            raise ValueError("rnn_type should be either gru or lstm")
        self.rnn = rnn(
            embedding_dim + hidden_dim, hidden_dim, n_layers, dropout_p
        )
        self.w_s = nn.Linear(hidden_dim, vocab_size)
        self.attention = Attention(hidden_dim)

    def init_weight(self):
        #  nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.w_s.weight)
        self.rnn.init_weight()

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        encoder_lengths: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ):
        """
        Args:
            inputs (torch.Tensor): A (batch, seq_len) tensor, within which
                each sequence is a list of token IDs
            hidden (torch.Tensor): A (batch, num_layers, num_direction * hidden_size)
                tensor containing the hidden state of the last time step
            encoder_outputs (torch.Tensor): A (batch, seq_len, hidden_size) tensor
                containing the outputs of the encoder
            encoder_lengths (torch.Tensor): The source sequence length
                ``(batch,)``.

        """
        use_teacher_forcing = (
            True if random.random() < teacher_forcing_ratio else False
        )

        # the <sos> tag
        emb_inputs = self.embedding(inputs[:, 0])
        emb_inputs = self.dropout(emb_inputs)
        max_length = inputs.shape[1] - 1
        attn_h = torch.zeros(inputs.shape[0], self.hidden_dim).to(
            inputs.device
        )

        decode = []
        for i in range(max_length):
            hidden, attn_h, softmaxed = self.step(
                attn_h, emb_inputs, hidden, encoder_outputs, encoder_lengths
            )
            #  output, hidden = self.rnn(
            #  torch.cat([attn_h, emb_inputs], dim=1), hidden
            #  )
            #  attn_h = self.attention(output, encoder_outputs, encoder_lengths)
            #  softmaxed = F.log_softmax(self.w_s(attn_h), 1)
            decode.append(softmaxed)

            # Get predicted word
            if use_teacher_forcing:
                idx = inputs[:, i + 1]
            else:
                idx = softmaxed.max(1)[1]

            emb_inputs = self.embedding(idx)
            emb_inputs = self.dropout(emb_inputs)

        scores = torch.cat(decode, 1)
        return scores.reshape(inputs.shape[0] * max_length, -1)

    def step(
        self, attn_h, emb_inputs, hidden, encoder_outputs, encoder_lengths
    ):
        output, hidden = self.rnn(
            torch.cat([attn_h, emb_inputs], dim=1), hidden
        )
        attn_h = self.attention(output, encoder_outputs, encoder_lengths)
        softmaxed = F.log_softmax(self.w_s(attn_h), 1)

        return hidden, attn_h, softmaxed


# testing
if __name__ == "__main__":
    N_EPOCH = 50
    BATCH_SIZE = 64
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 100
    N_LAYERS = 3
    LR = 0.001
    MAX_LENGTH = 20

    from eng2freDataset import Dataset
    from preprocess import transform

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ds = Dataset("./data/eng-fra.txt", transform=transform)
    data_loader = ds.get_loader(BATCH_SIZE, True)
    train_X, train_y, len_X, len_y = list(iter(data_loader))[0]
    decoder = nn.DataParallel(
        Decoder(
            ds.get_target_vocab_size(),
            EMBEDDING_DIM,
            HIDDEN_DIM * 2,
            N_LAYERS,
            0.5,
            "lstm",
            0.5,
        ).to(device),
        dim=0,
    )
    preds = decoder(
        train_y,
        tuple(
            [
                torch.randn(BATCH_SIZE, N_LAYERS, 2 * HIDDEN_DIM).to(device)
                for _ in range(2)
            ]
        ),
        torch.randn(BATCH_SIZE, train_X.shape[1], 2 * HIDDEN_DIM).to(device),
        torch.randint(1, train_X.shape[1] + 1, (train_X.shape[0],)),
    )
    print(preds.shape)
