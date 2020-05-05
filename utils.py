import torch 
import numpy as np
from sklearn.metrics import pairwise_distances

def get_topic_diversity(beta, topk):

    logits = pairwise_distances(beta, metric='cosine')
    TD = logits[np.triu_indices(logits.shape[0], k = 1)].mean()
    print('Topic diveristy is: {}'.format(TD))

    return TD

def get_document_frequency(data, wi, wj=None):

    if wj is None:

        D_wi = 0

        for l in range(len(data)):
            doc = data[l].squeeze(0)
            if wi in doc:
                D_wi += 1

        return D_wi

    D_wj = 0
    D_wi_wj = 0

    for l in range(len(data)):
        doc = data[l].squeeze(0)
        if wj in doc:
            D_wj += 1
            if wi in doc:
                D_wi_wj += 1

    return D_wj, D_wi_wj 

def get_topic_coherence(beta, data, vocab):

    D = len(data) ## number of docs...data is list of documents
    print('D: ', D)
    TC = []
    num_topics = len(beta)

    for k in range(num_topics):

        print('k: {}/{}'.format(k, num_topics))
        top_10 = list(beta[k].argsort()[-11:][::-1])
        top_words = [vocab[a] for a in top_10]

        TC_k = 0
        counter = 0
        for i, word in enumerate(top_10):

            D_wi = get_document_frequency(data, word)
            j = i + 1
            tmp = 0

            while j < len(top_10) and j > i:

                D_wj, D_wi_wj = get_document_frequency(data, word, top_10[j])
                f_wi_wj = np.log(D_wi_wj + 1) - np.log(D_wi)
                tmp += f_wi_wj
                j += 1
                counter += 1

            TC_k += tmp 
        TC.append(TC_k / counter)

    TC = np.mean(TC)
    print('Topic coherence is: {}'.format(TC))

    return TC

def nearest_neighbors(word, embeddings, vocab):
    vectors = embeddings.data.cpu().numpy() 
    index = vocab.index(word)
    print('vectors: ', vectors.shape)
    query = vectors[index]
    print('query: ', query.shape)
    ranks = vectors.dot(query).squeeze()
    denom = query.T.dot(query).squeeze()
    denom = denom * np.sum(vectors**2, 1)
    denom = np.sqrt(denom)
    ranks = ranks / denom
    mostSimilar = []
    [mostSimilar.append(idx) for idx in ranks.argsort()[::-1]]
    nearest_neighbors = mostSimilar[:20]
    nearest_neighbors = [vocab[comp] for comp in nearest_neighbors]
    return nearest_neighbors
