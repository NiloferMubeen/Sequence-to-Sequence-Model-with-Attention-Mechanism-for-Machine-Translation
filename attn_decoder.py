import torch
import torch.nn as nn
from attention import BahdanauAttention

class AttentionDecoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, attention, dropout=0.1):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, embed_dim)

        self.rnn = nn.GRU(
            embed_dim + hidden_dim,
            hidden_dim
        )

        self.fc_out = nn.Linear(
            embed_dim + hidden_dim * 2,
            output_dim
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        """
        input: (batch_size)
        hidden: (1, batch_size, hidden_dim)
        encoder_outputs: (src_len, batch_size, hidden_dim)
        """

        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        # embedded: (1, batch_size, embed_dim)

        decoder_hidden = hidden[-1]
        # (batch_size, hidden_dim)

        context, attention_weights = self.attention(
            decoder_hidden, encoder_outputs
        )
        # context: (batch_size, hidden_dim)

        context = context.unsqueeze(0)
        # (1, batch_size, hidden_dim)

        rnn_input = torch.cat((embedded, context), dim=2)
        # (1, batch_size, embed_dim + hidden_dim)

        output, hidden = self.rnn(rnn_input, hidden)
        # output: (1, batch_size, hidden_dim)

        output = output.squeeze(0)
        embedded = embedded.squeeze(0)
        context = context.squeeze(0)

        prediction = self.fc_out(
            torch.cat((output, context, embedded), dim=1)
        )
        # (batch_size, output_dim)

        return prediction, hidden, attention_weights
