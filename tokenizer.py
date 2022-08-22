def _preprocess(s):
    s = s.lower()
    s = s.replace("\s+"," ")
    return s
class Tokenizer:
    def __init__(self, tokenize = lambda s: s.split() ):
        self.tokenize = tokenize
        self.vocab_size = 0
        self.tokens = {}
    def fit_text(self,text):
        sequences = []
        for sentence in text:
            sentence = _preprocess(sentence)
            for token in set (self.tokenize(sentence)):
                if token not in self.tokens:
                    self.vocab_size += 1
                    self.tokens[token] = self.vocab_size




    def _tokenize_string(self,sentence):
        sequence = []
        for token in self.tokenize(sentence):
            if token in self.tokens:
                sequence.append(self.tokens[token])
        return sequence
    def text_to_sequence(self,text):
        if isinstance(text,str):
            return self._tokenize_string(text)
        sequences = []
        for sentence in text:
            sentence = _preprocess(sentence)
            sequence = self._tokenize_string(sentence)
            sequences.append(sequence)

        return sequences
if __name__  == "__main__":
    tokenizer = Tokenizer()
    tokenizer.fit_text(["the girl is fat"])
    print(tokenizer.text_to_sequence("the girl is fat"))
