import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import re
import nltk

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage

from utils.get_text_from_file import read_text_from_file
from utils.lemmatizator import lemmatize

wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')


def preproc_doc(doc):
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I | re.A)
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

    bw = CountVectorizer(min_df=0., max_df=1.)
    bw_matrix = bw.fit_transform(p_corpus).toarray()

    vocab_sorted = bw.get_feature_names_out()
    pd.DataFrame(bw_matrix, columns=vocab_sorted).to_csv("bagOfWords.csv", sep='\t', index=False)

    tfidf = TfidfTransformer(norm='l2', use_idf=True)
    tfidf_matrix = tfidf.fit_transform(bw_matrix).toarray()

    pd.DataFrame(np.round(tfidf_matrix, 2), columns=vocab_sorted).to_csv("tfidf.csv", sep='\t', index=False)

    similarity_matrix = cosine_similarity(tfidf_matrix)
    similarity_df = pd.DataFrame(similarity_matrix)
    similarity_df.to_csv("similarity_df.csv", index=False)

    ag = AgglomerativeClustering(n_clusters=3, metric='euclidean',linkage='ward')
    ag.fit(similarity_df)
    labels = ag.labels_
    df_labels = pd.DataFrame({'Document': range(len(labels)), 'Cluster': labels})
    df_labels.to_csv("clusters.csv", index=False)

    for i, (label, text) in enumerate(zip(labels, corpus)):
        print(f"Doc {i} belongs to Cluster {label}: {text}")

    links = linkage(similarity_matrix, 'complete')

    df_communication_matrix = pd.DataFrame(links, columns=['Document\Cluster 1', 'Document\Cluster 2','Distance', 'Cluster Size'])
    df_communication_matrix.to_csv("communication_matrix.csv", sep='\t', index=False)

    plt.figure(figsize=(8, 3))
    plt.title('Dendrogram')
    plt.xlabel('Document')
    plt.ylabel('Length')
    dendrogram(links)
    plt.show()
