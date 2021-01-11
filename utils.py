import torch 
from torch import optim
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

def visualize(m, show_emb=True):
    if not os.path.exists('./results'):
        os.makedirs('./results')

    m.eval()

    queries = ['andrew', 'computer', 'sports', 'religion', 'man', 'love', 
                'intelligence', 'money', 'politics', 'health', 'people', 'family']

    ## visualize topics using monte carlo
    with torch.no_grad():
        print('#'*100)
        print('Visualize topics...')
        topics_words = []
        gammas = m.get_beta().detach().cpu().numpy()
        for k in range(args.num_topics):
            gamma = gammas[k]
            top_words = list(gamma.argsort()[-args.num_words+1:][::-1])
            topic_words = [vocab[a] for a in top_words]
            topics_words.append(' '.join(topic_words))
            print('Topic {}: {}'.format(k, topic_words))

        if show_emb:
            ## visualize word embeddings by using V to get nearest neighbors
            print('#'*100)
            print('Visualize word embeddings by using output embedding matrix')
            try:
                embeddings = m.rho.weight  # Vocab_size x E
            except:
                embeddings = m.rho         # Vocab_size x E

            print('#'*100)

def get_optimizer(args, config, model):

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['wdecay'])
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=config['lr'], weight_decay=args.wdecay)
    elif args.optimizer == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=config['lr'], weight_decay=args.wdecay)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=config['lr'], weight_decay=args.wdecay)
    elif args.optimizer == 'asgd':
        optimizer = optim.ASGD(model.parameters(), lr=config['lr'], t0=0, lambd=0., weight_decay=args.wdecay)
    else:
        print('Defaulting to vanilla SGD')
        optimizer = optim.SGD(model.parameters(), lr=config['lr'])

    return optimizer
