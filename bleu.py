from sacrebleu import corpus_bleu
from translate import translate_sentence

def calculate_bleu(
    model,
    data,
    src_vocab,
    trg_vocab,
    device,
    max_len=50
):
    model.eval()

    references = []
    hypotheses = []

    for src, trg in data:
        src = src.to(device)
        trg = trg.to(device)

        for i in range(src.shape[1]):
            src_sentence = [
                src_vocab.idx2word[idx.item()]
                for idx in src[:, i]
                if idx.item() not in (
                    src_vocab.word2idx["<pad>"],
                    src_vocab.word2idx["<sos>"],
                    src_vocab.word2idx["<eos>"]
                )
            ]

            trg_sentence = [
                trg_vocab.idx2word[idx.item()]
                for idx in trg[:, i]
                if idx.item() not in (
                    trg_vocab.word2idx["<pad>"],
                    trg_vocab.word2idx["<sos>"],
                    trg_vocab.word2idx["<eos>"]
                )
            ]

            src_text = " ".join(src_sentence)
            trg_text = " ".join(trg_sentence)

            pred_tokens, _ = translate_sentence(
                model,
                src_text,
                src_vocab,
                trg_vocab,
                device,
                max_len
            )

            pred_text = " ".join(
                t for t in pred_tokens if t != "<eos>"
            )

            references.append([trg_text])
            hypotheses.append(pred_text)

    bleu = corpus_bleu(hypotheses, references)
    return bleu.score
