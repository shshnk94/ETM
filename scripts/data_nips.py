from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import pandas as pd
import pickle
import random
from scipy import sparse
import itertools
from scipy.io import savemat, loadmat
import re
import string
import sys
import os
import argparse

parser = argparse.ArgumentParser(description='nips preprocessing')

parser.add_argument('--folds', type=int, default=3, help='number of folds')
parser.add_argument('--path_save', type=str, help='directory containing data')
parser.add_argument('--split', type=float, help='percentage of test data')

args = parser.parse_args()
path_save = args.path_save

# Maximum / minimum document frequency
max_df = 0.7
min_df = 100  # choose desired value for min_df

# Read stopwords
with open('stops.txt', 'r') as f:
    stops = f.read().split('\n')

# Read data
print('reading text file...')
data_file = '../../data/papers.csv'
init_docs = list(pd.read_csv(data_file)['paper_text'])

init_docs = [re.findall(r'''[\w']+|[.,!?;-~{}`´_<=>:/@*()&'$%#"]''', init_docs[doc]) for doc in range(len(init_docs))]

def contains_punctuation(w):
    return any(char in string.punctuation for char in w)

def contains_numeric(w):
    return any(char.isdigit() for char in w)
    
# Removes all words with any punctuation or digits in them.
init_docs = [[w.lower() for w in init_docs[doc] if not contains_punctuation(w)] for doc in range(len(init_docs))]
init_docs = [[w for w in init_docs[doc] if not contains_numeric(w)] for doc in range(len(init_docs))]
init_docs = [[w for w in init_docs[doc] if len(w)>1] for doc in range(len(init_docs))]
init_docs = [" ".join(init_docs[doc]) for doc in range(len(init_docs))]

# Create count vectorizer
print('counting document frequency of words...')
cvectorizer = CountVectorizer(min_df=min_df, max_df=max_df, stop_words=None)
cvz = cvectorizer.fit_transform(init_docs).sign()

# Get vocabulary
print('building the vocabulary...')
sum_counts = np.asarray(cvz.sum(axis=0))[0]
v_size = sum_counts.shape[0]
print('initial vocabulary size: {}'.format(v_size))

# Sort elements in vocabulary and also remove stop words from the list
vocab = sorted([(i, word) for i, word in enumerate(cvectorizer.vocabulary_) if word  not in stops], key=lambda x: x[0])
vocab = [w for i, w in vocab]
del cvectorizer

# Split in train/test/valid
print('tokenizing documents and splitting into train/test/valid...')
tsSize = int(len(init_docs) * args.split)
num_docs_tr = trSize = len(init_docs) - tsSize
vaSize = int(num_docs_tr * args.split)
num_docs_tr = trSize = num_docs_tr - vaSize

#idx_permute = np.random.permutation(num_docs_tr).astype(int)
idx_permute = np.arange(num_docs_tr)

# Remove words not in train_data
vocab = list(set([w for idx_d in range(trSize) for w in init_docs[idx_permute[idx_d]].split() if w in vocab]))
word2id = dict([(w, j) for j, w in enumerate(vocab)])
id2word = dict([(j, w) for j, w in enumerate(vocab)])
print('vocabulary after removing words not in train: {}'.format(len(vocab)))

# Split in train/test/valid
docs_tr = [[word2id[w] for w in init_docs[idx_permute[idx_d]].split() if w in word2id] for idx_d in range(trSize)]
docs_va = [[word2id[w] for w in init_docs[idx_d+num_docs_tr].split() if w in word2id] for idx_d in range(vaSize)]
docs_ts = [[word2id[w] for w in init_docs[idx_d+num_docs_tr+vaSize].split() if w in word2id] for idx_d in range(tsSize)]

print('number of documents (train): {} [this should be equal to {}]'.format(len(docs_tr), trSize))
print('number of documents (test): {} [this should be equal to {}]'.format(len(docs_ts), tsSize))
print('number of documents (valid): {} [this should be equal to {}]'.format(len(docs_va), vaSize))

# Remove empty documents
print('removing empty documents...')

def remove_empty(in_docs):
    return [doc for doc in in_docs if doc!=[]]

docs_tr = remove_empty(docs_tr)
docs_ts = remove_empty(docs_ts)
docs_va = remove_empty(docs_va)

# Remove test documents with length=1
docs_ts = [doc for doc in docs_ts if len(doc)>1]

# Split test set in 2 halves
print('splitting test documents in 2 halves...')
docs_ts_h1 = [[w for i,w in enumerate(doc) if i<=len(doc)/2.0-1] for doc in docs_ts]
docs_ts_h2 = [[w for i,w in enumerate(doc) if i>len(doc)/2.0-1] for doc in docs_ts]
docs_va_h1 = [[w for i,w in enumerate(doc) if i<=len(doc)/2.0-1] for doc in docs_va]
docs_va_h2 = [[w for i,w in enumerate(doc) if i>len(doc)/2.0-1] for doc in docs_va]

# Get doc indices
print('getting doc indices...')

def create_doc_indices(in_docs):
    aux = [[j for i in range(len(doc))] for j, doc in enumerate(in_docs)]
    return [int(x) for y in aux for x in y]

doc_indices_tr = create_doc_indices(docs_tr)
doc_indices_ts_h1 = create_doc_indices(docs_ts_h1)
doc_indices_ts_h2 = create_doc_indices(docs_ts_h2)
doc_indices_va_h1 = create_doc_indices(docs_va_h1)
doc_indices_va_h2 = create_doc_indices(docs_va_h2)

print('len(np.unique(doc_indices_tr)): {} [this should be {}]'.format(len(np.unique(doc_indices_tr)), len(docs_tr)))
print('len(np.unique(doc_indices_ts_h1)): {} [this should be {}]'.format(len(np.unique(doc_indices_ts_h1)), len(docs_ts_h1)))
print('len(np.unique(doc_indices_ts_h2)): {} [this should be {}]'.format(len(np.unique(doc_indices_ts_h2)), len(docs_ts_h2)))
print('len(np.unique(doc_indices_va_h1)): {} [this should be {}]'.format(len(np.unique(doc_indices_va_h1)), len(docs_va_h1)))
print('len(np.unique(doc_indices_va_h2)): {} [this should be {}]'.format(len(np.unique(doc_indices_va_h2)), len(docs_va_h2)))

# Number of documents in each set
n_docs_tr = len(docs_tr)
n_docs_ts_h1 = len(docs_ts_h1)
n_docs_ts_h2 = len(docs_ts_h2)
n_docs_va_h1 = len(docs_va_h1)
n_docs_va_h2 = len(docs_va_h2)

# Create bow representation
print('creating bow representation...')

def create_list_words(in_docs):
    return [x for y in in_docs for x in y]

def create_bow(doc_indices, words, n_docs, vocab_size):
    return sparse.coo_matrix(([1]*len(doc_indices),(doc_indices, words)), shape=(n_docs, vocab_size)).tocsr()

bow_tr = create_bow(doc_indices_tr, create_list_words(docs_tr), n_docs_tr, len(vocab))
bow_ts_h1 = create_bow(doc_indices_ts_h1, create_list_words(docs_ts_h1), n_docs_ts_h1, len(vocab))
bow_ts_h2 = create_bow(doc_indices_ts_h2, create_list_words(docs_ts_h2), n_docs_ts_h2, len(vocab))
bow_va_h1 = create_bow(doc_indices_va_h1, create_list_words(docs_va_h1), n_docs_va_h1, len(vocab))
bow_va_h2 = create_bow(doc_indices_va_h2, create_list_words(docs_va_h2), n_docs_va_h2, len(vocab))

# Remove unused variables
del docs_tr
del docs_ts_h1
del docs_ts_h2
del docs_va_h1
del docs_va_h2
del doc_indices_tr
del doc_indices_ts_h1
del doc_indices_ts_h2
del doc_indices_va_h1
del doc_indices_va_h2

# Write the vocabulary to a file
if not os.path.isdir(path_save):
    os.system('mkdir -p ' + path_save)

with open(path_save + 'vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)
del vocab

# Split bow intro token/value pairs
print('splitting bow intro token/value pairs and saving to disk...')

def split_bow(bow_in, n_docs):
    indices = [[w for w in bow_in[doc,:].indices] for doc in range(n_docs)]
    counts = [[c for c in bow_in[doc,:].data] for doc in range(n_docs)]
    return indices, counts

bow_tr_tokens, bow_tr_counts = split_bow(bow_tr, n_docs_tr)
savemat(path_save + 'bow_tr_tokens.mat', {'tokens': bow_tr_tokens}, do_compression=True)
savemat(path_save + 'bow_tr_counts.mat', {'counts': bow_tr_counts}, do_compression=True)
del bow_tr
del bow_tr_tokens
del bow_tr_counts

bow_ts_h1_tokens, bow_ts_h1_counts = split_bow(bow_ts_h1, n_docs_ts_h1)
savemat(path_save + 'bow_ts_h1_tokens.mat', {'tokens': bow_ts_h1_tokens}, do_compression=True)
savemat(path_save + 'bow_ts_h1_counts.mat', {'counts': bow_ts_h1_counts}, do_compression=True)
del bow_ts_h1
del bow_ts_h1_tokens
del bow_ts_h1_counts

bow_ts_h2_tokens, bow_ts_h2_counts = split_bow(bow_ts_h2, n_docs_ts_h2)
savemat(path_save + 'bow_ts_h2_tokens.mat', {'tokens': bow_ts_h2_tokens}, do_compression=True)
savemat(path_save + 'bow_ts_h2_counts.mat', {'counts': bow_ts_h2_counts}, do_compression=True)
del bow_ts_h2
del bow_ts_h2_tokens
del bow_ts_h2_counts

bow_va_h1_tokens, bow_va_h1_counts = split_bow(bow_va_h1, n_docs_va_h1)
savemat(path_save + 'bow_va_h1_tokens.mat', {'tokens': bow_va_h1_tokens}, do_compression=True)
savemat(path_save + 'bow_va_h1_counts.mat', {'counts': bow_va_h1_counts}, do_compression=True)
del bow_va_h1
del bow_va_h1_tokens
del bow_va_h1_counts

bow_va_h2_tokens, bow_va_h2_counts = split_bow(bow_va_h2, n_docs_va_h2)
savemat(path_save + 'bow_va_h2_tokens.mat', {'tokens': bow_va_h2_tokens}, do_compression=True)
savemat(path_save + 'bow_va_h2_counts.mat', {'counts': bow_va_h2_counts}, do_compression=True)
del bow_va_h2
del bow_va_h2_tokens
del bow_va_h2_counts
print('Data ready !!')
