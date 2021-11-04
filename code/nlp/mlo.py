from dataset import load_data
from baseline import base_classify, load_leaves
from dataloader import create_dataset, get_dataloader
from trainer import MloClassifier, train, evaluate
import pandas as pd
from tqdm import tqdm
import random
import numpy as np
import torch
import pprint

import argparse
import copy
pp = pprint.PrettyPrinter()

def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_hyperparameter_dict(args):
    d = {
        'flags': {
            'feature': (args.feature_dim > 0),
            'electra': (args.electra_dim > 0),
            'liwc': (args.liwc_dim > 0),
            'vader': (args.vader_dim > 0),
            'liwc_leaves': (args.liwc_leaves_dim > 0),
            'vader_leaves': (args.vader_leaves_dim > 0),
            'final': (args.final_dim > 0),
        },
        'input_dim': {
        },
        'rep_dim': {
            'feature': args.feature_dim,
            'electra': args.electra_dim,
            'liwc': args.liwc_dim,
            'vader': args.vader_dim,
            'liwc_leaves': args.liwc_leaves_dim,
            'vader_leaves': args.vader_leaves_dim,
            'final': args.final_dim
        },
        'model_param': {
            'num_layers': args.num_layers,
            'dropout': args.dropout,
            'dim_features': args.dim_features,
            'activation': args.activation,
            'use_attention': args.use_attention > 0,
            'electra_from_pretrained': args.electra_from_pretrained
        },
        'training_param': {
            'seed': args.seed,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'gamma': args.gamma,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'num_epochs': args.num_epochs,
        }
    }

    return d

# Desired Rep Dims
#'feature': 16,
#'electra': 128,
#'liwc': 4,
#'vader': 4,
#'liwc_leaves': 8,
#'vader_leaves': 8,


@torch.no_grad()
def get_embeddings(train_dataset, valid_dataset, test_datset, param_dict):
    model = MloClassifier(param_dict)
    model = model.cuda()
    
    rd = param_dict['rep_dim']
    mp = param_dict['model_param']
    tp = param_dict['training_param']
    save_str = 'model'
    save_str += f'_feature_dim_{rd["feature"]}'
    save_str += f'_electra_dim_{rd["electra"]}'
    save_str += f'_liwc_dim_{rd["liwc"]}'
    save_str += f'_vader_dim{rd["vader"]}'
    save_str += f'_liwc_leaves_dim_{rd["liwc_leaves"]}'
    save_str += f'_vader_leaves_dim_{rd["vader_leaves"]}'
    save_str += f'_final_dim_{rd["final"]}'
    save_str += f'_use_attention_{mp["use_attention"]}'
    save_str += f'_dropout_{mp["dropout"]}'
    save_str += f'_seed_{tp["seed"]}'
    save_str += f'_learning_rate_{tp["learning_rate"]}'
    save_str += f'_batch_size_{tp["batch_size"]}'
    save_str += f'_weight_decay_{tp["weight_decay"]}'
    save_str += f'_num_epochs_{tp["num_epochs"]}'
    model.load_state_dict(torch.load(f'./saved_models/{save_str}'))
    
    train_dataloader = get_dataloader(train_dataset, shuffle=False, param_dict=param_dict)
    valid_dataloader = get_dataloader(valid_dataset, shuffle=False, param_dict=param_dict)
    test_dataloader = get_dataloader(test_dataset, shuffle=False, param_dict=param_dict)
    
    train_embeddings = []
    for d in train_dataloader:
        inputs, _ = d
        inputs = [i.cuda() for i in inputs]
        _, embeddings = model(inputs)
        train_embeddings.append(embeddings)
    train_embeddings = torch.cat(train_embeddings, dim=0)

    valid_embeddings = []
    for d in valid_dataloader:
        inputs, _ = d
        inputs = [i.cuda() for i in inputs]
        _, embeddings = model(inputs)
        valid_embeddings.append(embeddings)
    valid_embeddings = torch.cat(valid_embeddings, dim=0)

    test_embeddings = []
    for d in test_dataloader:
        inputs, _ = d
        inputs = [i.cuda() for i in inputs]
        _, embeddings = model(inputs)
        test_embeddings.append(embeddings)
    test_embeddings = torch.cat(test_embeddings, dim=0)
    
    save_str = save_str[6:]

    torch.save(train_embeddings, f'./saved_embeddings/train_z_{save_str}.pt')
    torch.save(valid_embeddings, f'./saved_embeddings/valid_z_{save_str}.pt')
    torch.save(test_embeddings, f'./saved_embeddings/test_z_{save_str}.pt')
    
def run_train(seed, train_dataset, valid_dataset, test_dataset, param_dict):
    reset_seed(seed)
    training_param_dict = param_dict['training_param']

    train_dataloader = get_dataloader(train_dataset, shuffle=True, param_dict=param_dict)
    valid_dataloader = get_dataloader(valid_dataset, shuffle=False, param_dict=param_dict)
    test_dataloader = get_dataloader(test_dataset, shuffle=False, param_dict=param_dict)

    model = MloClassifier(param_dict)
    print(model)
    model = model.cuda()

    criterion = torch.nn.BCEWithLogitsLoss()
    criterion = criterion.cuda()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_param_dict['learning_rate'],
        weight_decay=training_param_dict['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)

    optimizer.zero_grad()

    max_auc = -1
    max_model = None
    for epoch in tqdm(range(training_param_dict['num_epochs'])):
        model, optimizer, scheduler, train_loss = train(model, optimizer, scheduler, criterion, train_dataloader)
        #print(f'{epoch}th Epoch Train Loss: {train_loss}')
        eval_loss, pred_auc, prob_auc = evaluate(model, criterion, valid_dataloader, test=False)
        #print(f'{epoch}th Epoch Valid Loss: {eval_loss}, Pred AUC: {pred_auc}, Prob AUC: {prob_auc}')

        if pred_auc > max_auc:
            max_auc = pred_auc
            max_model = copy.deepcopy(model)

    eval_loss, pred_auc, prob_auc = evaluate(max_model, criterion, test_dataloader, test=True)
    print(f'Evaluation on Test Dataset: Pred AUC: {pred_auc:.04f}, Prob AUC: {prob_auc:.04f}')

    rd = param_dict['rep_dim']
    mp = param_dict['model_param']
    tp = param_dict['training_param']
    save_str = 'model'
    save_str += f'_feature_dim_{rd["feature"]}'
    save_str += f'_electra_dim_{rd["electra"]}'
    save_str += f'_liwc_dim_{rd["liwc"]}'
    save_str += f'_vader_dim{rd["vader"]}'
    save_str += f'_liwc_leaves_dim_{rd["liwc_leaves"]}'
    save_str += f'_vader_leaves_dim_{rd["vader_leaves"]}'
    save_str += f'_final_dim_{rd["final"]}'
    save_str += f'_use_attention_{mp["use_attention"]}'
    save_str += f'_dropout_{mp["dropout"]}'
    save_str += f'_seed_{tp["seed"]}'
    save_str += f'_learning_rate_{tp["learning_rate"]}'
    save_str += f'_batch_size_{tp["batch_size"]}'
    save_str += f'_weight_decay_{tp["weight_decay"]}'
    save_str += f'_num_epochs_{tp["num_epochs"]}'
    torch.save(model.state_dict(), f'./saved_models/{save_str}')

    return pred_auc, prob_auc, max_model

def _parse_args():
    parser = argparse.ArgumentParser()
    # Training Parameters
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', default=0.0000, type=float, help='Weight decay for optimizer')
    parser.add_argument('--gamma', default=0.98, type=float, help='Gamma for scheduler')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers for training')
    parser.add_argument('--num_epochs', default=20, type=int, help='Number of epochs for training')
    parser.add_argument('--seed', default=0, type=int, help='Number of epochs for training')
    # Embedding Dimensions
    parser.add_argument('--feature_dim', default=0, type=int, help='Embedding dimension for meta features. If 0, we do not use meta features.')
    parser.add_argument('--electra_dim', default=2, type=int, help='Embedding dimension for electra email embeddings. If 0, we do not use electra email embeddings.')
    parser.add_argument('--liwc_dim', default=2, type=int, help='Embedding dimension for liwc. If 0, we do not use liwc.')
    parser.add_argument('--vader_dim', default=2, type=int, help='Embedding dimension for vader. If 0, we do not use vader.')
    parser.add_argument('--liwc_leaves_dim', default=2, type=int, help='Embedding dimension for leaves from xgboost trained on liwc. If 0, we do not use liwc leaves.')
    parser.add_argument('--vader_leaves_dim', default=2, type=int, help='Embedding dimension for leaves from xgboost trained on vader. If 0, we do not use vader leaves.')
    parser.add_argument('--final_dim', default=0, type=int, help='Final embedding dimension')
    # Model Parameters
    parser.add_argument('--num_layers', default=2, type=int, help='Number of layers in the model')
    parser.add_argument('--use_attention', default=1, type=int, help='Option for using MHSA for email embeddings')
    parser.add_argument('--electra_from_pretrained', default=0, type=int, help='Option for using unsup-finetuned electra')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout of the model')
    parser.add_argument('--dim_features', default=256, type=int, help='Internal hidden dimension of the model')
    parser.add_argument('--activation', default='relu', type=str, help='Activation function of the model')
    
    parser.add_argument('--test', action='store_true', help='')
    parser.add_argument('--force_create', action='store_true', help='')

    return parser.parse_args()

if __name__=="__main__":
    # Prepare Data
    args = _parse_args()
    if args.force_create:
        XGB_LEAVE_LOAD = False
    else:
        XGB_LEAVE_LOAD = True

    raw_data, feature_data, liwc_embedding, vader_embedding, electra_embedding = load_data(force_create=args.force_create, electra_from_pretrained=args.electra_from_pretrained)
    P_train, P_valid, P_test, \
    X_out_train, X_out_valid, X_test, \
    X_in_train, X_in_valid, X_in_test, \
    y_train, y_valid, y_test = raw_data

    labels = (y_train, y_valid, y_test)

    if XGB_LEAVE_LOAD:
        liwc_leaves = load_leaves(name='liwc')
        vader_leaves = load_leaves(name='vader')
    else:
        print('\nliwc xgboost baseline')
        liwc_leaves = base_classify(liwc_embedding, labels, name='liwc')
        print('\nvader xgboost baseline')
        vader_leaves = base_classify(vader_embedding, labels, name='vader')

    # Set HyperParameters
    total_embedding_dim = sum([args.feature_dim, args.electra_dim, args.liwc_dim, args.vader_dim, args.liwc_leaves_dim, args.vader_leaves_dim])
    if total_embedding_dim==0:
        print('You must input at least one feature dim for training')
        assert(0)
    print(args)
    print(f'Total Embedding Dimension: {total_embedding_dim}')
    param_dict = set_hyperparameter_dict(args)
        
    param_dict['input_dim']['feature'] = feature_data[0][0].shape[0]

    param_dict['input_dim']['liwc'], = liwc_embedding[0][0].shape
    param_dict['input_dim']['vader'], = vader_embedding[0][0].shape

    param_dict['input_dim']['liwc_leaves'] = liwc_leaves[0][0].shape[0] * liwc_leaves[0][0].shape[1]
    param_dict['input_dim']['vader_leaves'] = vader_leaves[0][0].shape[0] * vader_leaves[0][0].shape[1]

    param_dict['input_dim']['electra'] = electra_embedding[0][0].shape[1]
    print('\nParameters:')
    pp.pprint(param_dict)

    # Get Dataset and DataLoader
    train_dataset, valid_dataset, test_dataset = create_dataset(
        (feature_data, liwc_embedding, vader_embedding, electra_embedding, liwc_leaves, vader_leaves, labels),
    )

    if args.test:
        get_embeddings(train_dataset, valid_dataset, test_dataset, param_dict)        
    else:
        pred_auc, prob_auc, max_model = run_train(args.seed, train_dataset, valid_dataset, test_dataset, param_dict)
