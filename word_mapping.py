from typing import List
import pandas as pd
from utils import pad_sequence, preprocess


class WordMapping:
    def __init__(self, all_captions_path, tokenize=lambda s: s.split()):
        self.tokenize = tokenize
        self.vocab_size = 1
        self.tokens = {"": 0}
        img_captions_df = pd.read_csv(all_captions_path)
        all_captions = img_captions_df['caption'].tolist()
        self.fit_text(all_captions)

    def fit_text(self, text: List[str]):
        sequences = []
        for sentence in text:
            sentence = preprocess(sentence)
            for token in set(self.tokenize(sentence)):
                if token not in self.tokens:
                    self.tokens[token] = self.vocab_size
                    self.vocab_size += 1

    def token_to_word(self, token):
        word_tokens_map = {val: key for key, val in self.tokens.items()}
        return word_tokens_map[token]

    def sequence_to_text(self,seq):
        result = " ".join([self.token_to_word(token) for token in seq])
        return result

    def tokenize_string(self, sentence):
        sequence = []
        for token in self.tokenize(sentence):
            if token in self.tokens:
                sequence.append(self.tokens[token])
        return sequence
