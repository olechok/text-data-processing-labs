import re

from gensim.models import word2vec
import nltk

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
    corpus = read_text_from_file("doc4.txt").split("\n")
    print("Original corpus:", corpus)

    p_corpus = [preproc_doc(doc) for doc in corpus]
    print("Processed corpus:", p_corpus)

    feature_size = 100
    window_context = 30
    min_word_count = 1
    sample = 1e-3

    tokenized_corpus = [wpt.tokenize(document) for document in p_corpus]
    w2v_model = word2vec.Word2Vec(tokenized_corpus,
                                  vector_size=feature_size,
                                  window=window_context,
                                  min_count=min_word_count,
                                  sample=sample)

    similar_words = {}
    for search_term in ['profit', 'dough']:
        if search_term in w2v_model.wv:
            similar_words[search_term] = [item[0] for item in w2v_model.wv.most_similar([search_term], topn=5)]
        else:
            similar_words[search_term] = "Word not found in vocabulary"

    print(similar_words)
