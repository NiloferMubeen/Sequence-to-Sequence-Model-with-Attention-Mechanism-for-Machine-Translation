import torch
import torch.nn as nn
import random

class AttnSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.rnn.hidden_size == decoder.rnn.hidden_size, \
            "Hidden dimensions must match"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: (src_len, batch_size)
        # trg: (trg_len, batch_size)

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        attentions = torch.zeros(trg_len, batch_size, src.shape[0]).to(self.device)

        encoder_outputs, hidden = self.encoder(src)

        input = trg[0]  # <sos>

        for t in range(1, trg_len):
            output, hidden, attn_weights = self.decoder(
                input, hidden, encoder_outputs
            )

            outputs[t] = output
            attentions[t] = attn_weights

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)

            input = trg[t] if teacher_force else top1

        return outputs, attentions
