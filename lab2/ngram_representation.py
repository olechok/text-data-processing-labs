import nltk
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from utils.get_text_from_file import read_text_from_file
from utils.lemmatizator import lemmatize


wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')


def preproc_doc(doc):
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower().strip()
    tokens = wpt.tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    doc = ' '.join(filtered_tokens)
    return lemmatize(doc)


if __name__ == "__main__":
    corpus = read_text_from_file("doc4.txt")
    corpus = corpus.split("\n")

    preproc_corpus = np.vectorize(preproc_doc)
    p_corpus = preproc_corpus(corpus)
    print(p_corpus)

    bv = CountVectorizer(ngram_range=(1, 2))
    bv_matrix = bv.fit_transform(p_corpus)
    bv_matrix = bv_matrix.toarray()
    vocab = bv.vocabulary_

    vocab_sorted = bv.get_feature_names_out()
    pd2 = pd.DataFrame(bv_matrix, columns=vocab_sorted)
    pd2.to_csv("table1.csv", sep='\t', index=False)

    word = "animal"
    if word in vocab:
        word_index = vocab[word]
        word_vector = bv_matrix[:, word_index]
        print(f"Vector for '{word}':", word_vector)
    else:
        print(f"Word '{word}' not in the dictionary.")
