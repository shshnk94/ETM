import numpy as np
import torch
from sklearn.metrics import pairwise_distances

def get_topic_diversity(beta):

    logits = 1 - torch.mm(beta, beta.transpose(0, 1)) / (torch.norm(beta, 2, 1) * torch.norm(beta, 2, 1))
    TD = logits[np.triu_indices(logits.shape[0], k = 1)].mean()
    print('Topic diveristy is: {}'.format(TD))

    return TD

def get_document_frequency(data, wi, wj=None, model='etm'):

    if wj is None:

        D_wi = 0

        for l in range(len(data)):

            D_wi += 1 if data[l][wi] else 0

        return D_wi

    D_wj = 0
    D_wi_wj = 0

    for l in range(len(data)):
        
        if True:
            doc = data[l]
            if doc[wj]:
                D_wj += 1
                if doc[wi]:
                    D_wi_wj += 1

        else:
            doc = data[l].keys() if isinstance(data[l], dict) else data[l].squeeze(0)
            if wj.item() in doc:
                D_wj += 1
                if wi.item() in doc:
                    D_wi_wj += 1

    return D_wj, D_wi_wj 

def get_topic_coherence(beta, data, model):

    D = len(data)
    TC = []
    num_topics = len(beta)

    for k in range(num_topics):

        #top_10 = list(beta[k].argsort()[-10:][::-1])
        top_10 = beta[k].argsort(descending=True)[:10]

        TC_k = 0
        counter = 0
        for i, word in enumerate(top_10):

            D_wi = get_document_frequency(data, word, model=model)
            j = i + 1
            tmp = 0

            while j < len(top_10) and j > i:

                D_wj, D_wi_wj = get_document_frequency(data, word, top_10[j], model=model)
                f_wi_wj = (np.log(D_wi_wj + 1) + np.log(D) - np.log(D_wi) - np.log(D_wj)) / (np.log(D) - np.log(D_wi_wj + 1))

                tmp += f_wi_wj
                j += 1
                counter += 1

            TC_k += tmp 

        TC.append(TC_k / counter)

    TC = np.mean(TC)
    print('Topic coherence is: {}'.format(TC))

    return TC

def get_perplexity(corpus, theta, beta):

    num_words = corpus.sum(dim=1)
    probabilites = torch.log(torch.mm(theta, beta) + 1e-12)

    #https://github.com/ray-project/ray/blob/master/python/ray/tune/trial.py#L487 - metric should be instance of Number.
    perplexity = torch.exp(-(probabilites * corpus).sum() / num_words.sum()).item()
    return perplexity
