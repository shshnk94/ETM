#/usr/bin/python
from __future__ import print_function

import sys
import argparse
import torch
import pickle 
import numpy as np 
import os 
import math 
import random 
import matplotlib.pyplot as plt 
import data
import scipy.io

from torch import nn, optim
from torch.nn import functional as F

from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from metrics import get_topic_coherence, get_topic_diversity, get_perplexity

from etm import ETM
from utils import nearest_neighbors
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='The Embedded Topic Model')

### data and file related arguments
parser.add_argument('--dataset', type=str, default='20ng', help='name of corpus')
parser.add_argument('--data_path', type=str, default='data/20ng', help='directory containing data')
parser.add_argument('--emb_path', type=str, default='data/20ng_embeddings.txt', help='directory containing word embeddings')
parser.add_argument('--save_path', type=str, default='./results', help='path to save results')
parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for training')

### model-related arguments
parser.add_argument('--num_topics', type=int, default=50, help='number of topics')
parser.add_argument('--rho_size', type=int, default=300, help='dimension of rho')
parser.add_argument('--emb_size', type=int, default=300, help='dimension of embeddings')
parser.add_argument('--t_hidden_size', type=int, default=800, help='dimension of hidden space of q(theta)')
parser.add_argument('--theta_act', type=str, default='relu', help='tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu)')
parser.add_argument('--train_embeddings', type=int, default=0, help='whether to fix rho or train it')

### optimization-related arguments
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--lr_factor', type=float, default=4.0, help='divide learning rate by this...')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train...150 for 20ng 100 for others')
parser.add_argument('--mode', type=str, default='train', help='train or eval model')
parser.add_argument('--optimizer', type=str, default='adam', help='choice of optimizer')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 1)')
parser.add_argument('--enc_drop', type=float, default=0.0, help='dropout rate on encoder')
parser.add_argument('--clip', type=float, default=0.0, help='gradient clipping')
parser.add_argument('--nonmono', type=int, default=10, help='number of bad hits allowed')
parser.add_argument('--wdecay', type=float, default=1.2e-6, help='some l2 regularization')
parser.add_argument('--anneal_lr', type=int, default=0, help='whether to anneal the learning rate or not')
parser.add_argument('--bow_norm', type=int, default=1, help='normalize the bows or not')

### evaluation, visualization, and logging-related arguments
parser.add_argument('--num_words', type=int, default=10, help='number of words for topic viz')
parser.add_argument('--log_interval', type=int, default=2, help='when to log training')
parser.add_argument('--visualize_every', type=int, default=10, help='when to visualize results')
parser.add_argument('--eval_batch_size', type=int, default=1000, help='input batch size for evaluation')
parser.add_argument('--load_from', type=str, default='', help='the name of the ckpt to eval from')
parser.add_argument('--tc', type=int, default=0, help='whether to compute topic coherence or not')
parser.add_argument('--td', type=int, default=0, help='whether to compute topic diversity or not')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('\n')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

## get data
# 1. vocabulary
vocab, train, valid, test = data.get_data(os.path.join(args.data_path))
vocab_size = len(vocab)
args.vocab_size = vocab_size

# 1. training data
train_tokens = train['tokens']
train_counts = train['counts']
args.num_docs_train = len(train_tokens)

# 2. dev set
valid_1_tokens = valid['tokens_1']
valid_1_counts = valid['counts_1']
valid_2_tokens = valid['tokens_2']
valid_2_counts = valid['counts_2']
args.num_docs_valid = len(valid_1_tokens)

# 3. test data
test_1_tokens = test['tokens_1']
test_1_counts = test['counts_1']
test_2_tokens = test['tokens_2']
test_2_counts = test['counts_2']
args.num_docs_test = len(test_1_tokens)

embeddings = None
if not args.train_embeddings:
    emb_path = args.emb_path
    vect_path = os.path.join(args.data_path.split('/')[0], 'embeddings.pkl')   
    vectors = {}
    with open(emb_path, 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            if word in vocab:
                vect = np.array(line[1:]).astype(np.float)
                vectors[word] = vect
    embeddings = np.zeros((vocab_size, args.emb_size))
    words_found = 0
    for i, word in enumerate(vocab):
        try: 
            embeddings[i] = vectors[word]
            words_found += 1
        except KeyError:
            embeddings[i] = np.random.normal(scale=0.6, size=(args.emb_size, ))
    embeddings = torch.from_numpy(embeddings).to(device)
    args.embeddings_dim = embeddings.size()

print('=*'*100)
print('Training an Embedded Topic Model on {} with the following settings: {}'.format(args.dataset.upper(), args))
print('=*'*100)

## define checkpoint
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

if args.mode == 'eval':
    ckpt = args.load_from
else:
    ckpt = os.path.join(args.save_path, 
        'etm_{}_K_{}_Htheta_{}_Optim_{}_Clip_{}_ThetaAct_{}_Lr_{}_Bsz_{}_RhoSize_{}_trainEmbeddings_{}'.format(
        args.dataset, args.num_topics, args.t_hidden_size, args.optimizer, args.clip, args.theta_act, 
            args.lr, args.batch_size, args.rho_size, args.train_embeddings))

## define model and optimizer
model = ETM(args.num_topics, vocab_size, args.t_hidden_size, args.rho_size, args.emb_size, 
                args.theta_act, embeddings, args.train_embeddings, args.enc_drop).to(device)

print('model: {}'.format(model))

if args.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'adagrad':
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'adadelta':
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'rmsprop':
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'asgd':
    optimizer = optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
else:
    print('Defaulting to vanilla SGD')
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

def train(epoch):

    model.train()

    acc_loss = 0
    acc_kl_theta_loss = 0

    cnt = 0
    indices = torch.randperm(args.num_docs_train)
    indices = torch.split(indices, args.batch_size)

    for idx, ind in enumerate(indices):

        optimizer.zero_grad()
        model.zero_grad()

        data_batch = data.get_batch(train_tokens, train_counts, ind, args.vocab_size, device)
        sums = data_batch.sum(1).unsqueeze(1)

        if args.bow_norm:
            normalized_data_batch = data_batch / sums
        else:
            normalized_data_batch = data_batch

        recon_loss, kld_theta = model(data_batch, normalized_data_batch)
        total_loss = recon_loss + kld_theta
        total_loss.backward()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        acc_loss += torch.sum(recon_loss).item()
        acc_kl_theta_loss += torch.sum(kld_theta).item()
        cnt += 1

        if idx % args.log_interval == 0 and idx > 0:
            print('Epoch: {}. batch: {}/{}. LR: {}. KL_theta: {}. Rec_loss: {}. NELBO: {}'.format(epoch, 
                                                                                                  idx, 
                                                                                                  len(indices), 
                                                                                                  optimizer.param_groups[0]['lr'], 
                                                                                                  round(acc_kl_theta_loss / cnt, 2),
                                                                                                  round(acc_loss / cnt, 2),
                                                                                                  round((acc_loss + acc_kl_theta_loss) / cnt, 2)))
    
    print('*'*100)
    print('Epoch: {}. batch: {}/{}. LR: {}. KL_theta: {}. Rec_loss: {}. NELBO: {}'.format(epoch, 
                                                                                          idx, 
                                                                                          len(indices), 
                                                                                          optimizer.param_groups[0]['lr'], 
                                                                                          round(acc_kl_theta_loss / cnt, 2),
                                                                                          round(acc_loss / cnt, 2),
                                                                                          round((acc_loss + acc_kl_theta_loss) / cnt, 2)))
    print('*'*100)
    return (acc_loss + acc_kl_theta_loss) / cnt

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
        gammas = m.get_beta()
        for k in range(args.num_topics):
            gamma = gammas[k]
            top_words = list(gamma.cpu().numpy().argsort()[-args.num_words+1:][::-1])
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
            """
            neighbors = []
            for word in queries:
                print('word: {} .. neighbors: {}'.format(
                    word, nearest_neighbors(word, embeddings, vocab)))
            """
            print('#'*100)

def report_score(model, writer, epoch, score, source):
    
    if writer is not None:
        for name, weights in model.named_parameters():
            writer.add_histogram(name, weights, epoch)
 
    with open(args.save_path + '/' + source + '_scores', 'a') as handle:
        handle.write(','.join([str(value) for key, value in score.items()]))

    return

def evaluate(m, source, writer=None, epoch=None, tc=False, td=False):

    """Compute perplexity on document completion.
    """
    m.eval()
    with torch.no_grad():
        if source == 'val':
            num_docs_test = args.num_docs_valid
            test_tokens_h1 = valid_1_tokens
            test_counts_h1 = valid_1_counts
            test_tokens_h2 = valid_2_tokens
            test_counts_h2 = valid_2_counts

        else: 
            num_docs_test = args.num_docs_test
            test_tokens_h1 = test_1_tokens
            test_counts_h1 = test_1_counts
            test_tokens_h2 = test_2_tokens
            test_counts_h2 = test_2_counts

        ## get \beta here
        beta = m.get_beta()
        
        theta = []
        indices = torch.split(torch.tensor(range(num_docs_test)), args.eval_batch_size)
        for idx, ind in enumerate(indices):

            ## get theta from first half of docs
            data_batch_1 = data.get_batch(test_tokens_h1, test_counts_h1, ind, args.vocab_size, device)
            sums_1 = data_batch_1.sum(1).unsqueeze(1)
            if args.bow_norm:
                normalized_data_batch_1 = data_batch_1 / sums_1
            else:
                normalized_data_batch_1 = data_batch_1

            theta.append(m.get_theta(normalized_data_batch_1)[0].cpu().detach().numpy())
        
        theta = np.concatenate(theta, axis=0)

        #Score to report
        score = {}

        ## get prediction loss using second half
        data_batch_2 = []
        for idx, ind in enumerate(indices):
            data_batch_2.append(data.get_batch(test_tokens_h2, test_counts_h2, ind, args.vocab_size, device).detach().cpu().numpy())

        data_batch_2 = np.concatenate(data_batch_2, axis=0)
        perplexity = get_perplexity(data_batch_2, theta, beta.detach().cpu().numpy())

        score['ppl'] = perplexity
        if tc or td:
            beta = beta.data.cpu().numpy()
            if tc:
                print('Computing topic coherence...')
                coherence = get_topic_coherence(beta, train_tokens, 'etm')
            if td:
                print('Computing topic diversity...')
                diversity = get_topic_diversity(beta, 'etm')

        score['tc'] = coherence if tc else np.nan
        score['td'] = diversity if td else np.nan
        report_score(m, writer, epoch, score, source)

        return perplexity

if args.mode == 'train':

    ## train model on data 
    best_epoch = 0
    best_val_ppl = 1e9
    all_val_ppls = []

    print('\n')
    print('Visualizing model quality before training...')
    visualize(model)
    print('\n')

    #Tensorboard writer
    if not os.path.exists(args.save_path + '/logs'):
        os.makedirs(args.save_path + '/logs')
    writer = SummaryWriter(args.save_path + '/logs/')

    for epoch in range(args.epochs):

        train_loss = train(epoch)
        val_ppl = evaluate(model, 'val', writer, epoch, args.tc, args.td)
        
        #Log loss into Tensorboard
        writer.add_scalar('Training Loss', train_loss, epoch)
        writer.add_scalar('Validation PPL', val_ppl, epoch)

        if val_ppl < best_val_ppl or not epoch:
            with open(ckpt, 'wb') as f:
                torch.save(model, f)
            best_epoch = epoch
            best_val_ppl = val_ppl

        else:
            ## check whether to anneal lr
            lr = optimizer.param_groups[0]['lr']
            if args.anneal_lr and (len(all_val_ppls) > args.nonmono and val_ppl > min(all_val_ppls[:-args.nonmono]) and lr > 1e-5):
                optimizer.param_groups[0]['lr'] /= args.lr_factor

        if epoch % args.visualize_every == 0:
            visualize(model)

        all_val_ppls.append(val_ppl)

    with open(ckpt, 'rb') as f:
        model = torch.load(f)

    model = model.to(device)
    val_ppl = evaluate(model, 'val', writer, args.epochs, args.tc, args.td)

else:   
    with open(ckpt, 'rb') as f:
        model = torch.load(f)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        ## get document completion perplexities
        test_ppl = evaluate(model, 'test', tc=args.tc, td=args.td)

        ## get most used topics
        indices = torch.tensor(range(args.num_docs_train))
        indices = torch.split(indices, args.batch_size)
        thetaAvg = torch.zeros(1, args.num_topics).to(device)
        thetaWeightedAvg = torch.zeros(1, args.num_topics).to(device)
        cnt = 0
        for idx, ind in enumerate(indices):
            data_batch = data.get_batch(train_tokens, train_counts, ind, args.vocab_size, device)
            sums = data_batch.sum(1).unsqueeze(1)
            cnt += sums.sum(0).squeeze().cpu().numpy()
            if args.bow_norm:
                normalized_data_batch = data_batch / sums
            else:
                normalized_data_batch = data_batch
            theta, _ = model.get_theta(normalized_data_batch)
            thetaAvg += theta.sum(0).unsqueeze(0) / args.num_docs_train
            weighed_theta = sums * theta
            thetaWeightedAvg += weighed_theta.sum(0).unsqueeze(0)
            if idx % 100 == 0 and idx > 0:
                print('batch: {}/{}'.format(idx, len(indices)))
        thetaWeightedAvg = thetaWeightedAvg.squeeze().cpu().numpy() / cnt

        print('\nThe 10 most used topics are {}'.format(thetaWeightedAvg.argsort()[::-1][:10]))
        ## show topics
        beta = model.get_beta()
        topic_indices = list(np.random.choice(args.num_topics, 10)) # 10 random topics
        print('\n')
        with open(args.save_path + '/topics.txt', 'w') as f:
            for k in range(args.num_topics):#topic_indices:
                gamma = beta[k]
                top_words = list(gamma.cpu().numpy().argsort()[-args.num_words+1:][::-1])
                topic_words = [vocab[a] for a in top_words]
                f.write('Topic {}: {}\n'.format(k, topic_words))
                print('Topic {}: {}'.format(k, topic_words))

        if args.train_embeddings:
            ## show etm embeddings 
            try:
                rho_etm = model.rho.weight.cpu()
            except:
                rho_etm = model.rho.cpu()
            queries = ['andrew', 'woman', 'computer', 'sports', 'religion', 'man', 'love', 
                            'intelligence', 'money', 'politics', 'health', 'people', 'family']
            print('\n')
            print('ETM embeddings...')
            #for word in queries:
            #    print('word: {} .. etm neighbors: {}'.format(word, nearest_neighbors(word, rho_etm, vocab)))
            print('\n')
