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
                    self.tokens[token] = self.vocab_size
                    self.vocab_size += 1




    def text_to_sequence(self,text):
        sequences = []
        for sentence in text:
            sentence = _preprocess(sentence)
            sequence = []
            for token in self.tokenize(sentence):
                if token in self.tokens:
                    sequence.append(self.tokens[token])
            sequences.append(sequence)
        return sequences
if __name__  == "__main__":
    tokenizer = Tokenizer()
    tokenizer.fit_text(["i am tall", "i am fat","my fat is mohamed"])
    print(tokenizer.text_to_sequence(["i fat", "my name is mohamed"]))
