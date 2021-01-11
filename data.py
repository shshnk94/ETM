import os
import random
import pickle
import numpy as np
import torch 
import scipy.io

def object_to_tensor(tokens, counts, vocab_size, device):

    data = torch.zeros(tokens.shape[0], vocab_size).float().to(device)

    for i in range(data.shape[0]):

        doc, count = tokens[i], counts[i]

        if len(doc) == 1: 
            doc = [doc.squeeze()]
            count = [count.squeeze()]
        else:
            doc = doc.squeeze()
            count = count.squeeze()

        for j, word in enumerate(doc):
            data[i, word] = torch.from_numpy(count[j]).float().to(device)

    return data

def _fetch(path, name, vocab_size, device):

    if name == 'train':
        token_file = os.path.join(path, 'bow_tr_tokens.mat')
        count_file = os.path.join(path, 'bow_tr_counts.mat')
        tokens = scipy.io.loadmat(token_file)['tokens'].squeeze()
        counts = scipy.io.loadmat(count_file)['counts'].squeeze()

        return object_to_tensor(tokens, counts, vocab_size, device)

    else:
        if name == 'valid':
            token_1_file = os.path.join(path, 'bow_va_h1_tokens.mat')
            count_1_file = os.path.join(path, 'bow_va_h1_counts.mat')
            token_2_file = os.path.join(path, 'bow_va_h2_tokens.mat')
            count_2_file = os.path.join(path, 'bow_va_h2_counts.mat')

        else:
            token_1_file = os.path.join(path, 'bow_ts_h1_tokens.mat')
            count_1_file = os.path.join(path, 'bow_ts_h1_counts.mat')
            token_2_file = os.path.join(path, 'bow_ts_h2_tokens.mat')
            count_2_file = os.path.join(path, 'bow_ts_h2_counts.mat')

        tokens_1 = scipy.io.loadmat(token_1_file)['tokens'].squeeze()
        counts_1 = scipy.io.loadmat(count_1_file)['counts'].squeeze()
        tokens_2 = scipy.io.loadmat(token_2_file)['tokens'].squeeze()
        counts_2 = scipy.io.loadmat(count_2_file)['counts'].squeeze()

        return object_to_tensor(tokens_1, counts_1, vocab_size, device), object_to_tensor(tokens_2, counts_2, vocab_size, device)

def get_data(path, step, device, fold=''):
    
    with open(os.path.join(path, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)

    if step == 'train':

        train = _fetch(os.path.join(path, 'fold{}'.format(fold)) if fold != '' else path, 'train', len(vocab), device)
        #valid_1, valid_2 = _fetch(os.path.join(path, 'fold{}'.format(fold)) if fold != '' else path, 'valid', len(vocab), device)

        return vocab, train#, valid_1, valid_2
    
    elif step == 'valid':
        valid_1, valid_2 = _fetch(os.path.join(path, 'fold{}'.format(fold)) if fold != '' else path, 'valid', len(vocab), device)
        return valid_1, valid_2

    else:
        test_1, test_2 = _fetch(path, 'test', len(vocab), device)
        return test_1, test_2

def get_batch(data, ind, device):
    return data[ind].to(device)
