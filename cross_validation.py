import argparse
import os 
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

parser = argparse.ArgumentParser(description='etm cross validation wrapper')

parser.add_argument('--dataset', type=str, default='20ng', help='name of corpus')
parser.add_argument('--topics', type=str, default='50', help='number of topics')
parser.add_argument('--data_path', type=str, help='path to a fold of data')
parser.add_argument('--save_path', type=str, help='save path for every run')
parser.add_argument('--gpu', type=str, help='index of the gpu core which would contain this model')

args = parser.parse_args()

if args.dataset == '20ng':
    data_path = 'data/etm/20ng'
else:
    data_path = 'ETM\/data\/nips'

train_embeddings = '1'

def run_script(params, fold):
    
    os.system('CUDA_VISIBLE_DEVICES={} python etm/main.py --mode train'.format(args.gpu) + 
                  ' --dataset ' + args.dataset +
                  ' --fold ' + str(fold) + 
                  ' --data_path ' + args.data_path +
                  ' --save_path ' + args.save_path +
                  ' --num_topics ' + args.topics +
                  ' --train_embeddings ' + train_embeddings +
                  ' --epochs ' + params['epochs'] +
                  ' --lr ' + params['lr'])#+ ' &')
    """
    os.system('CUDA_VISIBLE_DEVICES={} python etm/main.py --mode test'.format(gpu) + 
                  ' --dataset ' + args.dataset +
                  ' --fold ' + str(fold) + 
                  ' --data_path ' + args.data_path +
                  ' --save_path ' + args.save_path +
                  ' --num_topics ' + args.topics +
                  ' --train_embeddings ' + train_embeddings +
                  ' --lr ' + params['lr'] +
                  ' --tc ' + '1' + 
                  ' --td ' + '1' +
                  ' --load_from ' + os.path.join(args.save_path, 
                                                 'fold{}'.format(fold), 
                                                 'etm_{}_k_{}_e_{}'.format(args.dataset, args.topics, params['epochs']), 
                                                 'model.ckpt'))# + ' &')
    """

#Hyperparameters
hyperparameters = {'epochs': ['5000'],
                   'lr': ['5e-5', '5e-4', '5e-3']}

for params in ParameterGrid(hyperparameters):

    metrics = np.zeros((int(params['epochs']), 3))
    for fold in range(3): #Hard coded values of fold

        run_script(params, fold)

        path = os.path.join(args.save_path, 'k{}_e{}_lr{}'.format(args.topics, params['epochs'], float(params['lr'])), 'fold{}'.format(fold), 'val_scores.csv')
        df = pd.read_csv(path, header=None)
        metrics += df.values

    metrics /= 3

    min_epoch = np.argmin(metrics[:, 0])
    print('Best epoch: {} (for the learning rate {}) with average ppl of {}'.format(min_epoch + 1, params['lr'], metrics[min_epoch, 0]))
