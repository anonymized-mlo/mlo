import pandas as pd
import sqlite3 as sql
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import random
import pickle
import math
from matplotlib import pyplot as plt
import seaborn as sns
import os
import copy
import sys, argparse

import torch
from torch import nn, optim
import torch.nn.functional as functional
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

from transformers import AdamW, get_cosine_schedule_with_warmup


class TrainDataset(Dataset):
    def __init__(self, input_data):
        """
        TrainDataset Initializer
        
        Attributes:
        input_data -- Input data with email embeddings, features, and labels
        """
        self.n_feature = len(input_data)
        self.data = input_data
        
    def __len__(self):
        return len(self.data[0])
        
    def __getitem__(self, item):
        ret = []
        for d in self.data:
            ret.append(d[item])
        
        return ret

def create_data_loader(input_data, batch_size, num_workers, collate_fn, shuffle):
    """
    Create data loader with given parameter
    
    Arguments:
    input_data -- Input data with email embeddings, features, and labels
    batch_size -- The size of mini-batch for training
    num_workers -- The number of CPU workers in training
    collate_fn -- The collate function to process data from data loader
    
    Returns:
    TrainDataset object which returns a batch of data for each iteration
    """
    ds = TrainDataset(input_data)
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=collate_fn
    )

class Classifier(nn.Module):
    def __init__(
        self,
        dropout: float=0.1,
    ):

        super().__init__()
        self.classify = nn.Linear(BERT_HIDDEN * 2, 2)
        self.tanh = nn.Tanh()
        
    def pool(self, X):
        if POOLING == 'MEAN':
            result = torch.mean(X, 1)
        else:
            assert(0)
            
        return result
    
    def forward(
        self,
        X_in, X_out
    ):
        result_in = self.pool(X_in)
        result_out = self.pool(X_out)
        result = torch.cat([result_in, result_out], 1)
        result = self.classify(result)
        result = self.tanh(result)
        
        return result

def data_collate_fn(samples
):
    """
    The collate function for data loader
    
    Arguments:
    samples -- A batch of [X, F, y]
    
    Returns:
    X -- A batch of email embeddings after preprocessing
    F -- A batch of features after preprocessing
    y -- A batch of labels after preprocessing
    """
    
    # Exctract X, F, y from samples
    X_out_embedding, X_in_embedding, y = zip(*samples)
    
    X_out_embedding = pad_sequence(X_out_embedding, padding_value=0, batch_first=True)
    X_in_embedding = pad_sequence(X_in_embedding, padding_value=0, batch_first=True)
    
    y = torch.Tensor(y).to(torch.long)
    
    output = X_out_embedding, X_in_embedding, y
    return output


def train_and_test(classifier, X, y, run, seed, learning_rate, weight_decay, batch_size, epochs):
    # Training Stage
    # Please change parameters below if you want
    random.seed(seed)
    torch.manual_seed(seed)

    X_train, X_valid, X_test = X
    X_out_train_target, X_in_train_target = X_train
    X_out_valid_target, X_in_valid_target = X_valid
    X_out_test_target, X_in_test_target = X_test

    USE_SCHEDULER = True
    if MULTI_GPU:
        batch_size = batch_size * torch.cuda.device_count()
    num_workers = 32

    total_steps = math.ceil(len(X_out_train_target) / batch_size) * epochs
    warmup_steps = len(X_out_train_target) / batch_size

    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer.zero_grad()

    if USE_SCHEDULER:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps = warmup_steps,
            num_training_steps = total_steps
        )
    
    min_valid_loss = float('inf')
    loss_data = []


    for epoch in range(epochs):
        # Make data loader
        train_data_loader = create_data_loader((X_out_train_target, X_in_train_target,
                                                y_train),
                                               batch_size, num_workers, data_collate_fn, True)
        valid_data_loader = create_data_loader((X_out_valid_target, X_in_valid_target,
                                                y_valid),
                                               batch_size, num_workers, data_collate_fn, True)
        test_data_loader = create_data_loader((X_out_test_target, X_in_test_target,
                                               y_test),
                                              batch_size, num_workers, data_collate_fn, True)
    
        train_loss = []
        valid_loss = []
    
        classifier.train()
        n = 0

        for d in train_data_loader:
            n += 1
        
            # Get a batch of X, F, y and load on GPU
            X_out, X_in, y = d
        
            X_out = X_out.to(gpu)
            X_in = X_in.to(gpu)
        
            output = classifier(X_out, X_in)
            del X_out, X_in
        
            # Calculate the loss of the output and the label, and do backward propagation
            y = y.to(gpu)
            loss = loss_func(output, y)
            del output, y

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if USE_SCHEDULER:
                scheduler.step()
        
            # Save the loss of this step
            train_loss.append(loss.item())
            del loss

        avg_train_loss = sum(train_loss) / len(train_loss)
        loss_data.append([epoch, avg_train_loss, 'Train'])
        print(str(run)+'th run, '+str(epoch)+'th epoch, Avg Train Loss: '+str(avg_train_loss))
    
        # Start Validation
    
        classifier.eval()
        with torch.no_grad():
            predictions = []
            answers = []
            for d in valid_data_loader:
                # Get a batch of X, F, y and load on GPU
                X_out, X_in, y = d

                X_out = X_out.to(gpu)
                X_in = X_in.to(gpu)

                output = classifier(X_out, X_in)
                del X_out, X_in

                # Calculate the loss of the output and the label, and predict the answer
                y = y.to(gpu)
                loss = loss_func(output, y)            
                valid_loss.append(loss.item())
                del loss
            
                _, preds = torch.max(output, dim=1)
                predictions.extend(preds)
        
                answers.extend(y)
            
                del preds, y, output

        predictions = torch.stack(predictions).cpu().tolist()
        answers = torch.stack(answers).cpu().tolist()

        avg_valid_loss = sum(valid_loss) / len(valid_loss)
        loss_data.append([epoch, avg_valid_loss, 'Valid'])
        print(str(run)+'th run, '+str(epoch)+'th epoch, Avg Valid Loss: '+str(avg_valid_loss))
        
        auc = roc_auc_score(answers, predictions)
        print(str(run)+'th run, Valid AUC Score: '+str(auc)+'\n')
        sys.stdout.flush()
        
        if avg_valid_loss < min_valid_loss:
            torch.save(classifier.state_dict(), 'baseline_bert.bin')
            min_valid_loss = avg_valid_loss
        
    print('Minimum Valid Loss: ' + str(min_valid_loss))
    sys.stdout.flush()

    # Evaluation Starts
    classifier.load_state_dict(torch.load('baseline_bert.bin'))

    classifier.eval()
    test_loss = []

    with torch.no_grad():
        predictions = []
        answers = []
        for d in test_data_loader:
            # Get a batch of X, F, y and load on GPU
            X_out, X_in, y = d
        
            X_out = X_out.to(gpu)
            X_in = X_in.to(gpu)
        
            output = classifier(X_out, X_in)
            del X_out, X_in

            # Calculate the loss of the output and the label, and predict the answer
            y = y.to(gpu)
            loss = loss_func(output, y)
        
            test_loss.append(loss.item())
            del loss

            _, preds = torch.max(output, dim=1)
            predictions.extend(preds)
        
            answers.extend(y)
            del y, output, preds

    predictions = torch.stack(predictions).cpu().tolist()
    answers = torch.stack(answers).cpu().tolist()

    avg_test_loss = sum(test_loss) / len(test_loss)
    print(str(run)+'th run, Avg Test Loss: '+str(avg_test_loss))

    class_names = ['Negative', 'Positive']
    auc = roc_auc_score(answers, predictions)
    print(str(run)+'th run, Test AUC Score: '+str(auc)+'\n')

    print(classification_report(answers, predictions, target_names=class_names))

    classification_dict = classification_report(answers, predictions, target_names=class_names, output_dict=True)
    classification_dict['auc'] = auc
    sys.stdout.flush()

    return classification_dict

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-run', '-n', help='Number of runs', default=5, dest='n_run')
    parser.add_argument('--random-seed', '-r', help='Start Seed', default=123, dest='seed')
    parser.add_argument('--model', '-m', help='BERT Family Model', default='electra', dest='bert_type')
    parser.add_argument('--pooling', '-p', help='Pooling Method', default='mean', dest='pooling')

    n_run = int(parser.parse_args().n_run)
    start_seed = int(parser.parse_args().seed)
    bert_type = parser.parse_args().bert_type
    if parser.parse_args().pooling == 'mean':
        pooling = 'MEAN'
    else:
        assert(0)

    return n_run, start_seed, bert_type, pooling

if __name__ == "__main__":
    n_run, start_seed, bert_type, pooling = get_arguments()

    DATA_DIR = None # This should be private
    MODEL_DIR = None # This should be private

    MULTI_GPU = True
    BERT_HIDDEN = 768
    BERT_MAX_LEN = 512
    BERT_TYPE = bert_type
    POOLING = pooling
    
    X_out_train_target = pd.read_pickle(DATA_DIR + 'X_out_train_' + bert_type + '.pkl')
    X_in_train_target = pd.read_pickle(DATA_DIR + 'X_in_train_' + bert_type + '.pkl')
    X_out_valid_target = pd.read_pickle(DATA_DIR + 'X_out_valid_' + bert_type + '.pkl')
    X_in_valid_target = pd.read_pickle(DATA_DIR + 'X_in_valid_' + bert_type + '.pkl')
    X_out_test_target = pd.read_pickle(DATA_DIR + 'X_out_test_' + bert_type + '.pkl')
    X_in_test_target = pd.read_pickle(DATA_DIR + 'X_in_test_' + bert_type + '.pkl')

    y_train = pd.read_pickle(DATA_DIR + 'y_train.pkl')
    y_valid = pd.read_pickle(DATA_DIR + 'y_valid.pkl')
    y_test = pd.read_pickle(DATA_DIR + 'y_test.pkl')   


    class_names = ['negative', 'positive']

    gpu = torch.device('cuda')
    cpu = torch.device('cpu')

    loss_func = nn.CrossEntropyLoss(reduction='mean')
    loss_func = loss_func.to(gpu)

    X_train = X_out_train_target, X_in_train_target
    X_valid = X_out_valid_target, X_in_valid_target
    X_test = X_out_test_target, X_in_test_target

    X = X_train, X_valid, X_test
    y = y_train, y_valid, y_test

    learning_rate = 1e-5
    weight_decay = 1e-7
    batch_size = 4
    epochs = 20

    seed = start_seed
    dict_list = []

    max_auc = 0
    max_model = None

    for i in range(n_run):
        classifier = Classifier()

        if MULTI_GPU:
            classifier = nn.DataParallel(classifier)
        classifier = classifier.to(gpu)

        d = train_and_test(classifier, X, y, i, seed, learning_rate, weight_decay, batch_size, epochs)
        dict_list.append(d)
        seed += 1

        if d['auc'] > max_auc:
            max_auc = d['auc']
            max_model = classifier
        else:
            del classifier

    total_dict = defaultdict(float)
    total_dict['Negative'] = defaultdict(float)
    total_dict['Positive'] = defaultdict(float)
    total_dict['macro avg'] = defaultdict(float)
    total_dict['weighted avg'] = defaultdict(float)

    for d in dict_list:
        for v_1 in d:
            if type(d[v_1]) is dict:
                for v_2 in d[v_1]:
                    total_dict[v_1][v_2] += d[v_1][v_2]
            else:
                total_dict[v_1] += d[v_1]

    for v_1 in total_dict:
        if type(total_dict[v_1]) is defaultdict:
            for v_2 in total_dict[v_1]:
                total_dict[v_1][v_2] = total_dict[v_1][v_2] / len(dict_list)
        else:
            total_dict[v_1] = total_dict[v_1] / len(dict_list)

    print('\n-----------------------------------')
    print('Final Result:')
    print(total_dict)
