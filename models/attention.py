import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import sequence_mask


class Attention(nn.Module):
    """
    Attention mechanics following Luong(2015) which takes a matrix and a
    query vector. It then computes a parameterized convex combination
    of the matrix based on the input query.

    Constructs a unit mapping a query `q` of size `hidden_dim`
    and a source matrix `H` of size `n x hidden_dim`, to an output
    of size `hidden_dim`.
    """

    def __init__(self, hidden_dim, dropout_p=0.2, attn_type="general"):
        super().__init__()
        assert attn_type in [
            "dot",
            "general",
        ], "Please select a valid attention type (got {:s}).".format(attn_type)

        self.attn_type = attn_type
        self.dropout = nn.Dropout(dropout_p)
        if self.attn_type == "general":
            self.w_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_c = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)

    def score(self, h_s: torch.Tensor, h_t: torch.Tensor):
        """
        Args:
            h_s (torch.Tensor): sequence of sources/encoder hidden states
                ```(batch, src_len, hidden_dim)```
            h_t (torch.Tensor): the query vector/decoder hidden states
                ```(batch, hidden_dim)```

        Returns:
            score (torch.Tensor): The attention score (not normalized)
                ```(batch, src_len)```
        """
        src_batch, src_len, src_dim = h_s.shape
        tgt_batch, tgt_dim = h_t.shape

        assert src_batch == tgt_batch
        assert src_dim == tgt_dim

        if self.attn_type == "general":
            # h_t_: ```(batch, d, 1)```
            h_t_ = self.w_a(h_t).unsqueeze(2)
        else:
            # h_t_: ```(batch, d, 1)```
            h_t_ = h_t.unsqueeze(2)

        # (batch, s_len, d) x (batch, d, 1) --> (batch, s_len, 1)
        score = torch.bmm(h_s, h_t_).squeeze(2)
        assert score.shape == (src_batch, src_len)
        return score

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_hidden: torch.Tensor,
        encoder_lengths: torch.Tensor,
    ):
        """
        Args:
            decoder_hidden (torch.Tensor): Query vector
                ``(batch, hidden_dim)``.
            encoder_hidden (torch.Tensor): Sequence of sources
                ``(batch, src_len, hidden_dim)``.
            encoder_lengths (torch.Tensor): The source sequence length
                ``(batch,)``.

        Returns:
            attn_h (torch.Tensor): The attentional hidden state
                ```(batch, src_len)```


        """
        tgt_batch, tgt_dim = decoder_hidden.shape
        src_batch, src_len, src_dim = encoder_hidden.shape

        assert src_batch == tgt_batch
        assert src_dim == tgt_dim

        # align_scores: (batch, src_len)
        align_scores = self.score(encoder_hidden, decoder_hidden)

        if encoder_lengths is not None:
            mask = sequence_mask(
                encoder_lengths, max_len=align_scores.shape[1]
            )
            align_scores.masked_fill_(1 - mask, -float("inf"))

        # align_vector:  (batch, src_len)
        align_vector = F.softmax(align_scores, dim=1)

        # (batch, 1, src_len) x (batch, src_len, hidden_dim)
        #  --> (batch, 1, hidden_dim)
        # context_vector: (batch, hidden_dim)
        context_vector = torch.bmm(
            align_vector.unsqueeze(1), encoder_hidden
        ).squeeze(1)

        # concat_c_h: (batch, 2 * hidden_dim)
        concat_c_h = torch.cat([context_vector, decoder_hidden], dim=1)

        # attentional hidden state: (batch, hidden_dim)
        attn_h = torch.tanh(self.w_c(concat_c_h))
        attn_h = self.dropout(attn_h)

        return attn_h
