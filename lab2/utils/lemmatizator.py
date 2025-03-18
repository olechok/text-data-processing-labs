from nltk import sent_tokenize, pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer


wlem = WordNetLemmatizer()

def map_pos_tag(pos_tag: str) -> str | None:
    mapping = {"V": "v", "N": "n", "J": "a", "R": "r"}

    for key, value in mapping.items():
        if pos_tag.startswith(key):
            return value
    return None


def map_word_with_pos(word_pos_tuple: tuple[str, str]) -> tuple[str, str]:
    word, pos = word_pos_tuple
    return word, map_pos_tag(pos)


def lemmatize(text: str) -> str:
    pos_tags = pos_tag(word_tokenize(text))
    wn_tagged = map(map_word_with_pos, pos_tags)
    res_words = []
    for word, tag in wn_tagged:
        if tag is None:
            res_words.append(word)
        else:
            res_words.append(wlem.lemmatize(word, tag))
    return " ".join(res_words)