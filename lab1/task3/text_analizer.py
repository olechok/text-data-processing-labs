from nltk.tokenize import regexp_tokenize, word_tokenize
from utils.get_text_from_file import read_text_from_file
from nltk.tokenize import sent_tokenize
from nltk import pos_tag
import pandas as pd
from nltk.stem import WordNetLemmatizer

wlem = WordNetLemmatizer()


def tokenize_without_numbers(text: str) -> list:
    pattern = r'\b[a-zA-Z]+\b'
    return regexp_tokenize(text, pattern)


def count_amount_of_words_in_3_sentence(text: str) -> int:
    sentences = sent_tokenize(text, language='english')
    words = regexp_tokenize(sentences[2], '\w+')
    return len(words)


def mark_parts_of_speech(text: str) -> list:
    pattern = r"\w+"
    words = regexp_tokenize(text, pattern)
    pos_tags = pos_tag(words)
    df = pd.DataFrame(pos_tags, columns=['WORD', 'POS TAG'])
    df.to_csv("pos_tagged.csv", sep='\t')
    return pos_tags


def derive_nouns(pos_tags: list) -> list:
    nouns = [(word, pos) for word, pos in pos_tags if pos in ['NN', 'NNS', 'NNP', 'NNPS']]
    return nouns


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
    sentences = sent_tokenize(text, language='english')
    pos_tags = pos_tag(word_tokenize(sentences[1]))
    wn_tagged = map(map_word_with_pos, pos_tags)
    res_words = []
    for word, tag in wn_tagged:
        if tag is None:
            res_words.append(word)
        else:
            res_words.append(wlem.lemmatize(word, tag))
    return " ".join(res_words)


if __name__ == '__main__':
    text = read_text_from_file("../text4.txt")

    # 1 task
    tokens = tokenize_without_numbers(text)
    print("Task1")
    print(tokens)
    with open("output.txt", "w", encoding="utf-8") as file:
        for item in tokens:
            file.write(f"{item}\n")

    # 2 task
    amount = count_amount_of_words_in_3_sentence(text)
    print("Task2")
    print(amount)

    # 3 task
    pos_tags = mark_parts_of_speech(text)
    nouns = derive_nouns(pos_tags)
    print("Task3")
    print(nouns)

    # 4 task
    lemmatized_text = lemmatize(text)
    print("Task4")
    print(lemmatized_text)
