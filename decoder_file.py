import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()

        self.output_dim = output_dim

        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.rnn = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        # input: (batch_size)
        # hidden: (num_layers, batch_size, hidden_dim)

        input = input.unsqueeze(0)
        # input: (1, batch_size)

        embedded = self.dropout(self.embedding(input))
        # embedded: (1, batch_size, embed_dim)

        output, hidden = self.rnn(embedded, hidden)
        # output: (1, batch_size, hidden_dim)

        prediction = self.fc_out(output.squeeze(0))
        # prediction: (batch_size, output_dim)

        return prediction, hidden
