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
import argparse
import sys

import torch
from torch import nn, optim
import torch.nn.functional as functional
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

from transformers import AdamW, get_cosine_schedule_with_warmup

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def get_valid_setting(start_seed, n_run, X_dict, y_dict, est_list, lr_list):
    X_train_ori = X_dict['train']
    X_valid = X_dict['valid']

    
    y_train_ori = y_dict['train']
    y_valid = y_dict['valid']

    
    final_auc = 0
    final_est = -1
    final_lr = -1
    final_model = None
    
    seed = start_seed
    for i in range(n_run):
        ros = RandomOverSampler(random_state=seed)

        X_train, y_train = ros.fit_sample(X=X_train_ori, y=y_train_ori)
        X_train = pd.DataFrame(data=X_train)
        X_valid = pd.DataFrame(data=X_valid)

        max_auc = 0
        max_est = None
        max_lr = None
        max_model = None

        for est in est_list:
            for lr in lr_list:
                xgb = XGBClassifier(n_estimators=est, learning_rate = lr, max_depth = 4)
                xgb.fit(X_train, y_train)
                xgb_pred = xgb.predict(X_valid)
                auc = roc_auc_score(y_valid, xgb_pred)

                if auc > max_auc:
                    max_auc = auc
                    max_est = est
                    max_lr = lr
                    max_model = xgb

        print("Valid MAX AUC:", max_auc, "Valid MAX EST:", max_est, "Valid MAX LR:", max_lr)
        sys.stdout.flush()
        if max_auc > final_auc:
            final_auc = max_auc
            final_est = max_est
            final_lr = max_lr
            final_model = max_model
            
        seed += 1
        
    print("\nValid Final AUC:", final_auc, "Valid Final EST:", final_est, "Valid MAX LR:", final_lr)
    sys.stdout.flush()
    return final_model, final_auc, final_est, final_lr


def test_xgboost(model, X_dict, y_dict):
    X_test = X_dict['test']
    y_test = y_dict['test']
    
    X_test = pd.DataFrame(data=X_test)
    xgb_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, xgb_pred)
    
    class_names = ['Negative', 'Positive']
    classification_dict = classification_report(y_test, xgb_pred, target_names=class_names, output_dict=True)

    print("---------------------------------")
    print("Test AUC:", auc)
    print(classification_report(y_test, xgb_pred, target_names=class_names))
    
    return auc, classification_dict

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-run', '-n', help='Number of runs', default=5, dest='n_run')
    parser.add_argument('--random-seed', '-r', help='Start Seed', default=123, dest='seed')

    n_run = int(parser.parse_args().n_run)
    start_seed = int(parser.parse_args().seed)

    return n_run, start_seed

if __name__ == "__main__":
    n_run, start_seed = get_arguments()

    DATA_DIR = None # This should be private.

    X_out_train_vader = pd.read_pickle(DATA_DIR + 'X_out_train_vader_aggregated.pkl')
    X_in_train_vader = pd.read_pickle(DATA_DIR + 'X_in_train_vader_aggregated.pkl')
    X_out_valid_vader = pd.read_pickle(DATA_DIR + 'X_out_valid_vader_aggregated.pkl')
    X_in_valid_vader = pd.read_pickle(DATA_DIR + 'X_in_valid_vader_aggregated.pkl')
    X_out_test_vader = pd.read_pickle(DATA_DIR + 'X_out_test_vader_aggregated.pkl')
    X_in_test_vader = pd.read_pickle(DATA_DIR + 'X_in_test_vader_aggregated.pkl')

    y_train = pd.read_pickle(DATA_DIR + 'y_train.pkl')
    y_valid = pd.read_pickle(DATA_DIR + 'y_valid.pkl')
    y_test = pd.read_pickle(DATA_DIR + 'y_test.pkl')
    
    assert(len(X_out_train_vader) == len(X_in_train_vader))
    assert(len(X_out_valid_vader) == len(X_in_valid_vader))
    assert(len(X_out_test_vader) == len(X_in_test_vader))

    X_train = []
    for i in range(len(X_out_train_vader)):
        x_out = X_out_train_vader[i].tolist()
        x_in = X_in_train_vader[i].tolist()
        x = x_out + x_in
        X_train.append(x)

    X_valid = []
    for i in range(len(X_out_valid_vader)):
        x_out = X_out_valid_vader[i].tolist()
        x_in = X_in_valid_vader[i].tolist()
        x = x_out + x_in
        X_valid.append(x)

    X_test = []
    for i in range(len(X_out_test_vader)):
        x_out = X_out_test_vader[i].tolist()
        x_in = X_in_test_vader[i].tolist()
        x = x_out + x_in
        X_test.append(x)
        
    X_dict = {'train': X_train, 'valid': X_valid, 'test': X_test}
    y_dict = {'train': y_train, 'valid': y_valid, 'test': y_test}
    
    est_list = [50, 100, 150]
    lr_list = [0.1, 0.05, 0.01]
    
    model, _, _, _ = get_valid_setting(start_seed, n_run, X_dict, y_dict, est_list, lr_list)
    test_xgboost(model, X_dict, y_dict)
    
    X_train_leaves = model.apply(pd.DataFrame(X_train))
    X_valid_leaves = model.apply(pd.DataFrame(X_valid))
    X_test_leaves = model.apply(pd.DataFrame(X_test))
    
    # One-hot Encoding
    nb_classes = 32
    one_hot_X_train_leaves = []
    for l in X_train_leaves:
        one_hot_X_train_leaves.append(torch.Tensor(np.eye(nb_classes)[l]))
    one_hot_X_valid_leaves = []
    for l in X_valid_leaves:
        one_hot_X_valid_leaves.append(torch.Tensor(np.eye(nb_classes)[l]))
    one_hot_X_test_leaves = []
    for l in X_test_leaves:
        one_hot_X_test_leaves.append(torch.Tensor(np.eye(nb_classes)[l]))
        
    with open(DATA_DIR + 'X_train_vader_xgboost_one_hot.pkl', 'wb') as f:
        pickle.dump(one_hot_X_train_leaves, f)
    with open(DATA_DIR + 'X_valid_vader_xgboost_one_hot.pkl', 'wb') as f:
        pickle.dump(one_hot_X_valid_leaves, f)
    with open(DATA_DIR + 'X_test_vader_xgboost_one_hot.pkl', 'wb') as f:
        pickle.dump(one_hot_X_test_leaves, f)
