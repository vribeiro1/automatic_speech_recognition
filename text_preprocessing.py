BLANK = "<blank>"


def text_processor(s, vocabulary):
    tokens = list(s)
    tokens_numerized = [vocabulary[token] for token in tokens]

    return tokens_numerized

