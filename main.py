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
from scipy.special import softmax

from torch import nn, optim
from torch.nn import functional as F
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest import ConcurrencyLimiter

from os import path
#sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from metrics import get_topic_coherence, get_topic_diversity, get_perplexity

from etm import ETM
from utils import nearest_neighbors, get_optimizer, visualize
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='The Embedded Topic Model')

### data and file related arguments
parser.add_argument('--fold', type=str, default='', help='current cross valid fold number')
#parser.add_argument('--valid', type=bool, default=True, help='toggle for training with validation vs final model')
parser.add_argument('--mode', type=str, default='hyperparam', help='choose between hp tuning, training, and evaluation.')
parser.add_argument('--dataset', type=str, default='20ng', help='name of corpus')
parser.add_argument('--data_path', type=str, default='data/20ng', help='directory containing data')
parser.add_argument('--emb_path', type=str, default='data/20ng_embeddings.txt', help='directory containing word embeddings')
parser.add_argument('--save_path', type=str, default='./results', help='path to save results')
parser.add_argument('--ckpt', type=str, help='path to saved checkpoint of the model')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training')

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
#parser.add_argument('--mode', type=str, default='train', help='train or eval model')
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
parser.add_argument('--td', type=int, default=0, help='whether to compute topic diversity or not')
parser.add_argument('--tc', type=int, default=0, help='whether to compute topic coherence or not')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('\n')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

## get data
# 1. vocabulary
"""
vocab, train_set, valid_1, valid_2 = data.get_data(os.path.join(args.data_path), 'train', device, args.fold)
vocab_size = len(vocab)
args.vocab_size = vocab_size
args.num_docs_train = train_set.shape[0]
args.num_docs_valid = valid_1.shape[0]
"""
embeddings = None
if not args.train_embeddings:

    print("Reading pretrained word embeddings...")
    with open(os.path.join(args.emb_path, 'embeddings.pkl'), 'rb') as f:
       embeddings = pickle.load(f)
 
    embeddings = torch.from_numpy(embeddings).to(device)
    args.embeddings_dim = embeddings.size()

print('=*'*100)
print('Training an Embedded Topic Model on {} with the following settings: {}'.format(args.dataset.upper(), args))
print('=*'*100)

"""
if args.mode == 'test':
    ckpt = args.load_from
else:
    if args.fold != '':
        ckpt = os.path.join(args.save_path, 'k{}_e{}_lr{}'.format(args.num_topics, args.epochs, args.lr), 'fold{}'.format(args.fold))
    else:
        ckpt = args.save_path
"""
## define checkpoint
#if not os.path.exists(ckpt):
#    os.makedirs(ckpt)

def step(train_set, model, optimizer, epoch):

    model.train()

    acc_loss = 0
    acc_kl_theta_loss = 0

    cnt = 0
    indices = torch.randperm(args.num_docs_train, device=device)
    indices = torch.split(indices, args.batch_size)

    for idx, ind in enumerate(indices):

        optimizer.zero_grad()
        #model.zero_grad()
        #torch.zero_grad()

        data_batch = data.get_batch(train_set, ind, device)
        normalized_data_batch = data_batch / data_batch.sum(1).unsqueeze(1) if args.bow_norm else data_batch

        recon_loss, kld_theta = model(data_batch, normalized_data_batch)
        total_loss = recon_loss + kld_theta
        total_loss.backward()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        acc_loss += torch.sum(recon_loss).item()
        acc_kl_theta_loss += torch.sum(kld_theta).item()
        cnt += 1

    print('*'*100)
    print('Epoch: {}. KL_theta: {}. Rec_loss: {}. NELBO: {}'.format(epoch, acc_kl_theta_loss / cnt, acc_loss / cnt, (acc_loss + acc_kl_theta_loss) / cnt))
    print('*'*100)

    return (acc_loss + acc_kl_theta_loss) / cnt

def report_score(model, writer, epoch, score, source):
     
    if writer is not None:
        for name, weights in model.named_parameters():
            writer.add_histogram(name, weights, epoch)
 
    with open(ckpt + '/' + source + '_scores.csv', 'a') as handle:
        handle.write(','.join([str(value) for key, value in score.items()]) + '\n')

    return

def test(m, source, writer=None, epoch=None):

    m.eval()
    with torch.no_grad():

        first_half, second_half = data.get_data(os.path.join(args.data_path), 'valid' if source == 'val' else 'test', device)
        num_docs_test = first_half.shape[0]

        ## get \beta here
        beta = m.get_beta()
        
        theta = []
        indices = torch.split(torch.tensor(range(num_docs_test), device=device), args.eval_batch_size)
        for idx, ind in enumerate(indices):

            ## get theta from first half of docs
            data_batch_1 = data.get_batch(first_half, ind, device)
            normalized_data_batch_1 = data_batch_1 / data_batch_1.sum(1).unsqueeze(1) if args.bow_norm else data_batch_1
            theta.append(m.get_theta(normalized_data_batch_1)[0])
        
        theta = torch.cat(theta, dim=0)

        #Score to report
        score = {}

        ## get prediction loss using second half
        data_batch_2 = []
        for idx, ind in enumerate(indices):
            data_batch_2.append(data.get_batch(second_half, ind, device))

        data_batch_2 = torch.cat(data_batch_2, dim=0)
        perplexity = get_perplexity(data_batch_2, theta, beta)

        score['ppl'] = perplexity
        if args.tc or args.td:

            #Store \beta during inference in the ckpt folder
            #with open(os.path.join(ckpt, 'beta.pkl'), 'wb') as f:
            #    pickle.dump(beta.detach().cpu().numpy(), f)

            if args.tc:

                print('Computing topic coherence...')
                coherence = get_topic_coherence(beta, train_set, 'etm')

            if args.td:

                print('Computing topic diversity...')
                diversity = get_topic_diversity(beta)

        score['tc'] = coherence if args.tc else np.nan
        score['td'] = diversity if args.td else np.nan
        #report_score(m, writer, epoch, score, source)
        
        return perplexity

def train(config, checkpoint_dir=None):
    
    #tune.utils.wait_for_gpu(1)
    vocab, train_set = data.get_data(os.path.join(args.data_path), 'train', device, args.fold)
    args.vocab_size = len(vocab)
    args.num_docs_train = train_set.shape[0]

    model = ETM(args.num_topics, 
                args.vocab_size, 
                config['t_hidden_size'], 
                config['rho_size'], 
                config['emb_size'], 
                args.theta_act, 
                embeddings, 
                args.train_embeddings, 
                config['enc_drop']).to(device)

    optimizer = get_optimizer(args, config, model)

    ## train model on data 
    #all_val_ppls = []

    #Tensorboard writer
    #if not os.path.exists(ckpt + '/logs'):
    #    os.makedirs(ckpt + '/logs')

    #FIXME: writer = SummaryWriter(ckpt + '/logs/')
    writer = None

    for epoch in range(args.epochs):

        train_loss = step(train_set, model, optimizer, epoch)
        val_ppl = test(model, 'val', writer, epoch)
        
        #Log loss into Tensorboard
        #writer.add_scalar('Validation PPL', val_ppl, epoch)
        #writer.add_scalar('Training Loss', train_loss, epoch)
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            torch.save((model.state_dict(), optimizer.state_dict()), os.path.join(checkpoint_dir, "checkpoint"))

        tune.report(validation_perplexity=val_ppl)
        #if val_ppl < best_val_ppl or not epoch:
        #if args.fold == "" and True:
        #    with open(ckpt + '/model.ckpt', 'wb') as f:
        #        torch.save(model, f)
        #    best_epoch = epoch
        #    best_val_ppl = val_ppl

        #else:
            ## check whether to anneal lr
        lr = optimizer.param_groups[0]['lr']
        if args.anneal_lr and (len(all_val_ppls) > args.nonmono and val_ppl > min(all_val_ppls[:-args.nonmono]) and lr > 1e-5):
            optimizer.param_groups[0]['lr'] /= args.lr_factor

        #if epoch % args.visualize_every == 0:
        #    visualize(model)

        #all_val_ppls.append(val_ppl)

    del model #to avoid Redis overflow as per https://github.com/ray-project/ray/issues/5439


def get_config(topics):

    if topics == 50:

        current_best = {'lr': 8e-4,
                        't_hidden_size': 800, 
                        'rho_size': 300, 
                        'emb_size': 300,
                        'enc_drop': 0.0,
                        'wdecay': 1.2e-6}

        space = {'lr': tune.loguniform(1e-5, 1e-3),
                 't_hidden_size': 800,#tune.choice(np.arange(100, 900, 100)), 
                 'rho_size': 300,#tune.choice(np.arange(100, 400, 100)), 
                 'emb_size': 300,#tune.choice(np.arange(100, 400, 100)),
                 'enc_drop': 0.0,#tune.loguniform(1e-6, 1.0),
                 'wdecay': tune.loguniform(5e-5, 5e-3)}

    return [current_best], space

if __name__ == '__main__':
    
    if args.mode == 'hyperparam':
         
        ray.init()
        current_best, space = get_config(args.num_topics) 

        algorithm = HyperOptSearch(metric="validation_perplexity", 
                                   mode="min",
                                   points_to_evaluate=current_best,
                                   #space=space,
                                   random_state_seed=0)
        algorithm = ConcurrencyLimiter(algorithm, max_concurrent=4)
        #print(algorithm.domain, algorithm._space)
        analysis = tune.run(train,
                            metric="validation_perplexity",
                            mode="min",
                            keep_checkpoints_num=1,
                            checkpoint_score_attr='min-validation_perplexity',
                            stop={"training_iteration": 2000},
                            resources_per_trial={"cpu": 2, "gpu": 0.33},
                            search_alg=algorithm,
                            num_samples=100,
                            config=space)

        print("Best config is:", analysis.best_config)
        with open(os.path.join(args.save_path, 'best_config.pkl'), 'wb') as f:
            pickle.dump(analysis.best_config, f)
    
    elif args.mode == 'train':
        
        with open(os.path.join(args.save_path, 'best_config.pkl'), 'rb') as f:
            config = pickle.load(f)
        
        scheduler = ASHAScheduler(time_attr='epochs',
                                  metric='validation_perplexity',
                                  mode='min',
                                  max_t=50,
                                  grace_period=50,
                                  reduction_factor=2,
                                  brackets=1)

        analysis = tune.run(train,
                            #metric="validation_perplexity",
                            #mode="min",
                            keep_checkpoints_num=1,
                            checkpoint_score_attr='min-validation_perplexity',
                            stop={"training_iteration": 2000},
                            resources_per_trial={"cpu": 2, "gpu": 0.25},
                            num_samples=1,
                            scheduler=scheduler,
                            config=config)
    
    else:

        vocab, train_set = data.get_data(os.path.join(args.data_path), 'train', device, args.fold)
        with open(os.path.join(args.save_path, 'best_config.pkl'), 'rb') as f:
            config = pickle.load(f)

        model = ETM(args.num_topics, 
                len(vocab), 
                config['t_hidden_size'], 
                config['rho_size'], 
                config['emb_size'], 
                args.theta_act, 
                embeddings, 
                args.train_embeddings, 
                config['enc_drop']).to(device)

        with open(args.ckpt, 'rb') as f:
            model.load_state_dict(torch.load(f)[0])

        with torch.no_grad():
            ## get document completion perplexities
            test_ppl = test(model, 'test')
            print("Document perplexity: ", test_ppl)

            ## get most used topics
            indices = torch.tensor(range(train_set.shape[0]))
            indices = torch.split(indices, args.batch_size)
            thetaAvg = torch.zeros(1, args.num_topics).to(device)
            thetaWeightedAvg = torch.zeros(1, args.num_topics).to(device)
            cnt = 0
            for idx, ind in enumerate(indices):
                data_batch = data.get_batch(train_set, ind, device)
                sums = data_batch.sum(1).unsqueeze(1)
                cnt += sums.sum(0).squeeze().cpu().numpy()
                if args.bow_norm:
                    normalized_data_batch = data_batch / sums
                else:
                    normalized_data_batch = data_batch
                theta, _ = model.get_theta(normalized_data_batch)
                thetaAvg += theta.sum(0).unsqueeze(0) / train_set.shape[0]
                weighed_theta = sums * theta
                thetaWeightedAvg += weighed_theta.sum(0).unsqueeze(0)
            thetaWeightedAvg = thetaWeightedAvg.squeeze().cpu().numpy() / cnt

            print('\nThe 10 most used topics are {}'.format(thetaWeightedAvg.argsort()[::-1][:10]))
            ## show topics
            beta = model.get_beta().detach().cpu().numpy()
            topic_indices = list(np.random.choice(args.num_topics, 10)) # 10 random topics
            print('\n')
            with open(args.save_path + '/topics.txt', 'w') as f:
                for k in range(args.num_topics):#topic_indices:
                    gamma = beta[k]
                    top_words = list(gamma.argsort()[-args.num_words+1:][::-1])
                    topic_words = [vocab[a] for a in top_words]
                    f.write('Topic {}: {}\n'.format(k, topic_words))
                    print('Topic {}: {}'.format(k, topic_words))
