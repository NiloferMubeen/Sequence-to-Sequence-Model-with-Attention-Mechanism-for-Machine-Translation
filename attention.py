import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        self.W_s = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        """
        decoder_hidden: (batch_size, hidden_dim)
        encoder_outputs: (src_len, batch_size, hidden_dim)
        """

        src_len = encoder_outputs.shape[0]

        # Repeat decoder hidden state src_len times
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        # (batch_size, src_len, hidden_dim)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # (batch_size, src_len, hidden_dim)

        energy = torch.tanh(
            self.W_h(encoder_outputs) + self.W_s(decoder_hidden)
        )
        # (batch_size, src_len, hidden_dim)

        scores = self.v(energy).squeeze(2)
        # (batch_size, src_len)

        attention_weights = F.softmax(scores, dim=1)
        # (batch_size, src_len)

        context = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_outputs
        ).squeeze(1)
        # (batch_size, hidden_dim)

        return context, attention_weights
