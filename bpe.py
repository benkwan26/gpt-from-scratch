import regex as re
from collections import defaultdict
from config import VOCAB_SIZE, PATTERN

def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))

class BPE:
    def __init__(self, text):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self._train(text)

    # ---------- BPE TRAINING ----------
    def _get_vocab(self, text):
        vocab = defaultdict(int)

        for token in re.findall(PATTERN, text):
            chars = ''.join(self.byte_encoder[b] for b in token.encode("utf-8"))
            vocab[tuple(chars)] += 1

        return vocab

    def _get_stats(self, vocab):
        pairs = defaultdict(int)

        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq

        return pairs

    def _merge_vocab(self, pair, vocab):
        new_vocab = {}

        for word, freq in vocab.items():
            new_word = []
            i = 0

            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            new_vocab[tuple(new_word)] = freq

        return new_vocab

    def _train(self, text):
        vocab = self._get_vocab(text)
        merges = []

        for _ in range(VOCAB_SIZE - 256):
            stats = self._get_stats(vocab)
            if not stats:
                break
            best = max(stats, key=stats.get)
            merges.append(best)
            vocab = self._merge_vocab(best, vocab)

        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}

        tokens = set()
        for word in vocab:
            tokens.update(word)

        self.token_to_id = {t: i for i, t in enumerate(tokens)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.vocab_size = len(self.token_to_id)

    # ---------- ENCODE / DECODE ----------
    def _bpe(self, token):
        word = tuple(token)
        pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}

        while True:
            pair = min(
                pairs,
                key=lambda p: self.bpe_ranks.get(p, float("inf")),
                default=None
            )
            if pair not in self.bpe_ranks:
                break

            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}

        return word

    def encode(self, text):
        ids = []

        for token in re.findall(PATTERN, text):
            chars = ''.join(self.byte_encoder[b] for b in token.encode("utf-8"))

            for t in self._bpe(chars):
                ids.append(self.token_to_id[t])

        return ids

    def decode(self, ids):
        text = ''.join(self.id_to_token[i] for i in ids)
        bytes_ = bytearray(self.byte_decoder[c] for c in text)

        return bytes_.decode("utf-8", errors="replace")