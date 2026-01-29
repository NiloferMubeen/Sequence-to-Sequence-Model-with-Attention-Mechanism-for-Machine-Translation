import torch
import torch.nn as nn



class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.rnn = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: (seq_len, batch_size)

        embedded = self.dropout(self.embedding(src))
        # embedded: (seq_len, batch_size, embed_dim)

        outputs, hidden = self.rnn(embedded)
        # outputs: (seq_len, batch_size, hidden_dim)
        # hidden:  (num_layers, batch_size, hidden_dim)

        return outputs, hidden
