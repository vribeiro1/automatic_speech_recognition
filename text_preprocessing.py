BLANK = "BLANK"
SILENCE = "SIL"
UNKNOWN = "UNK"


def text_processor(s, vocabulary):
    tokens = list(s)
    tokens_numerized = [vocabulary.get(token, vocabulary[UNKNOWN]) for token in tokens]
    return tokens_numerized
