import string
from collections import Counter

import re
import tensorflow as tf


class Tokenizer:
    def __init__(self, unk_threshold=5):
        self.unk_threshold = unk_threshold
        self.start, self.end, self.unk, self.pad = "<start>", "<end>", "<unk>", "<pad>"
        self.token_2_index = {self.pad: 0, self.unk: 1, self.start: 2, self.end: 3}
        self.index_2_token = {v: k for k, v in self.token_2_index.items()}

    @property
    def vocab_size(self):
        return len(self.token_2_index)

    @property
    def start_id(self):
        return self.token_2_index[self.start]

    @property
    def end_id(self):
        return self.token_2_index[self.end]

    @property
    def pad_id(self):
        return self.token_2_index[self.pad]

    def preprocess(self, text):
        t = text.lower()
        t = re.sub(f"([{string.punctuation}])", r" \1 ", t)
        t = re.sub(r"[^a-zA-Z.?!,;'\"]+", r" ", t)
        t = re.sub(r"\s+", r" ", t)
        t = t.strip()
        t = f"{self.start} {t} {self.end}"
        t = t.split(" ")
        return t

    def fit_on_texts(self, texts):
        texts = [self.preprocess(t) for t in texts]
        counter = Counter([token for text in texts for token in text])
        for token, count in counter.most_common():
            if count > self.unk_threshold and token not in self.token_2_index:
                index = self.vocab_size
                self.token_2_index[token] = index
                self.index_2_token[index] = token

    def text_to_sequence(self, text, max_len=None):
        sequence = text.numpy().decode("utf-8")
        sequence = self.preprocess(sequence)
        sequence = list(map(lambda token: self.token_2_index.get(token, self.token_2_index[self.unk]), sequence))
        if max_len is not None:
            sequence = sequence[:max_len]
        return tf.convert_to_tensor(sequence, dtype=tf.int64)

    def sequence_to_text(self, sequence):
        text = sequence.numpy()
        text = list(map(lambda index: self.index_2_token[index], text))
        text = " ".join(text)
        return tf.convert_to_tensor(text, dtype=tf.string)
