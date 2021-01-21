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

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

from transformers import AdamW, get_cosine_schedule_with_warmup

class EmailEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int=4,
        dropout: float=0.1,
    ):
        """
        Email Encoder Initializer
        
        Attributes:
        X-linear -- Linear function to project email representation(BERT_HIDDEN) to hidden_dim
        dropout -- Dropout Layer
        encoders -- Series of Transformer Encoders with number of num_layers
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=BERT_HIDDEN,
                                                   nhead=2,
                                                   dim_feedforward=512,
                                                   dropout=dropout,
                                                   activation='gelu')
        self.encoders = nn.TransformerEncoder(encoder_layer, num_layers=2)
    
    def forward(self, x):
        """
        Email Encoder Forward Function
        Make padding mask from x, and get transformer attention.
        
        Arguments:
        x -- input tensor
                in shape (batch_size, sequence_length, email_representation_length(BERT_HIDDEN))
                
        Returns:
        out -- output tensor
                in shape (batch_size, sequence_length, hidden_dim)
        """
        out = x
        
        # Get padding_mask of x
        pad = (x[:, :, 0] == 0.)
        
        # Get transformer encoding of x
        out = self.encoders(src=out.transpose(0,1), src_key_padding_mask=pad)
        out = out.transpose(0,1)
        
        del pad
        
        return out

class Classifier(nn.Module):
    def __init__(
        self,
        rep_dim: int=2,
        num_layers: int=4,
        dropout: float=0.1,
        dim_features: int=256,
    ):
        """
        Email Encoder Initializer
        
        Attributes:
        email_encoder -- Encoder layer which utilizes transformer
        email_classify -- Linear layer to project encoded email to 2-dim for classification
        
        F_linear -- Linear layer to project dimension of feature(1) to hidden_dim
        dropout -- Dropout Layer
        encoders -- Series of Transformer Encoders with number of num_layers
        final_classify -- Linear layer to project hidden_dim to 2-dim for classification
        """
        super().__init__()
        hidden_dim = BERT_HIDDEN
        self.rep_dim = rep_dim
        self.dim_features = dim_features
        
        self.total_rep_dim = (TRANSFORMER + VADER + VADER_LEAVES + LIWC + LIWC_LEAVES) * rep_dim
        
        # Email Transformer Embedding Layers
        if TRANSFORMER:
            self.email_out_encoder = EmailEncoder(num_layers=num_layers, dropout=dropout)        
            self.email_in_encoder = EmailEncoder(num_layers=num_layers, dropout=dropout)
            self.ED_linear = nn.Linear(2 * hidden_dim, rep_dim)
            self.dropout = nn.Dropout(dropout)
        
        # Vader Layers
        if VADER:
            self.vader_linear = nn.Linear(VADER_DIM * 2, rep_dim)
            self.dropout_vader_out = nn.Dropout(dropout)
            self.tanh_vader = nn.Tanh()

        # VADER_LEAVES Layers
        if VADER_LEAVES:
            self.vader_leaves_linear = nn.Linear(VADER_LEAVES_DIM, rep_dim)
            self.vader_leaves_dropout = nn.Dropout(dropout)
            
        # LIWC Layers
        if LIWC:
            self.liwc_linear = nn.Linear(LIWC_DIM * 2, rep_dim)
            self.dropout_liwc_out = nn.Dropout(dropout)
            self.tanh_liwc = nn.Tanh()
            
        # LIWC_LEAVES Layers
        if LIWC_LEAVES:
            self.liwc_leaves_linear = nn.Linear(LIWC_LEAVES_DIM, rep_dim)
            self.liwc_leaves_dropout = nn.Dropout(dropout)
        
        self.final_classify = nn.Linear(self.total_rep_dim, 2)
        self.activation = nn.Tanh()
        
    def get_transformer_embedding(
        self,
        email_out_embedding: torch.Tensor = None,
        email_in_embedding: torch.Tensor = None,
    ):
        # Email Transformer Embedding
        batch_size, sequence_length, hd = email_out_embedding.shape
        rep_email_out = self.email_out_encoder(email_out_embedding)
        rep_email_out = rep_email_out + email_out_embedding
        
        rep_email_in = self.email_in_encoder(email_in_embedding)
        rep_email_in = rep_email_in + email_in_embedding
        
        rep_email = torch.cat((torch.mean(rep_email_out, dim=1), torch.mean(rep_email_in, dim=1)), 1)
        email_rep = self.ED_linear(rep_email)
        email_rep = self.dropout(email_rep)
        
        return email_rep
    
    def get_vader_embedding(
        self,
        email_out_vader: torch.Tensor = None,
        email_in_vader: torch.Tensor = None,
    ):
        # Use Vader
        vader_rep = torch.cat([email_out_vader, email_in_vader], 1)
        vader_rep = self.vader_linear(vader_rep)
        vader_rep = self.dropout(vader_rep)
        
        return vader_rep

    def get_vader_leaves_embedding(
        self,
        email_vader_leaves: torch.Tensor = None,
    ):
        email_vader_leaves = torch.flatten(email_vader_leaves, start_dim=1)
        out = self.vader_leaves_linear(email_vader_leaves)
        out = self.vader_leaves_dropout(out)
        return out
    
    def get_liwc_embedding(
        self,
        email_out_liwc: torch.Tensor = None,
        email_in_liwc: torch.Tensor = None,
    ):
        # Use Vader
        liwc_rep = torch.cat([email_out_liwc, email_in_liwc], 1)
        liwc_rep = self.liwc_linear(liwc_rep)
        liwc_rep = self.dropout(liwc_rep)
        
        return liwc_rep
    
    def get_liwc_leaves_embedding(
        self,
        email_liwc_leaves: torch.Tensor = None,
    ):
        email_liwc_leaves = torch.flatten(email_liwc_leaves, start_dim=1)
        out = self.liwc_leaves_linear(email_liwc_leaves)
        out = self.liwc_leaves_dropout(out)
        return out
    
    def forward(
        self,
        email_out_embedding: torch.Tensor=None,
        email_in_embedding: torch.Tensor=None,
        email_out_vader: torch.Tensor = None,
        email_in_vader: torch.Tensor = None,
        email_vader_leaves: torch.Tensor = None,
        email_out_liwc: torch.Tensor = None,
        email_in_liwc: torch.Tensor = None,
        email_liwc_leaves: torch.Tensor = None,
    ):
        embeddings = []
        if TRANSFORMER:
            transformer_embedding = self.get_transformer_embedding(email_out_embedding, email_in_embedding)
            embeddings.append(transformer_embedding)
        if VADER:
            vader_embedding = self.get_vader_embedding(email_out_vader, email_in_vader)
            embeddings.append(vader_embedding)
        if VADER_LEAVES:
            vader_leaves_embedding = self.get_vader_leaves_embedding(email_vader_leaves)
            embeddings.append(vader_leaves_embedding)
        if LIWC:
            liwc_embedding = self.get_liwc_embedding(email_out_liwc, email_in_liwc)
            embeddings.append(liwc_embedding)
        if LIWC_LEAVES:
            liwc_leaves_embedding = self.get_liwc_leaves_embedding(email_liwc_leaves)
            embeddings.append(liwc_leaves_embedding)
        
        full_embedding = torch.cat(embeddings, dim=1)
        final_output = self.final_classify(full_embedding)
        final_output = self.activation(final_output)
            
        return final_output, full_embedding

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
    X_out_embedding, X_in_embedding, X_out_vader, X_in_vader, X_vader_leaves,                                 X_out_liwc, X_in_liwc, X_liwc_leaves, y = zip(*samples)
    
    X_out_embedding = pad_sequence(X_out_embedding, padding_value=0, batch_first=True)
    X_in_embedding = pad_sequence(X_in_embedding, padding_value=0, batch_first=True)
    X_out_vader = pad_sequence(X_out_vader, padding_value=0, batch_first=True)
    X_in_vader = pad_sequence(X_in_vader, padding_value=0, batch_first=True)
    X_vader_leaves = pad_sequence(X_vader_leaves, padding_value=0, batch_first=True)
    X_out_liwc = pad_sequence(X_out_liwc, padding_value=0, batch_first=True)
    X_in_liwc = pad_sequence(X_in_liwc, padding_value=0, batch_first=True)
    X_liwc_leaves = pad_sequence(X_liwc_leaves, padding_value=0, batch_first=True)
    y = torch.Tensor(y).to(torch.long)
    
    out = X_out_embedding, X_in_embedding, X_out_vader, X_in_vader, X_vader_leaves, X_out_liwc, X_in_liwc, X_liwc_leaves, y
    
    return out

def train_and_test(classifier, X, y, run, seed, learning_rate, weight_decay, batch_size, epochs):
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Training Stage
    # Please change parameters below if you want
    X_train, X_valid, X_test = X
    X_out_train_embedding, X_in_train_embedding, X_out_train_vader, X_in_train_vader, X_train_vader_leaves, \
        X_out_train_liwc, X_in_train_liwc, X_train_liwc_leaves = X_train
    X_out_valid_embedding, X_in_valid_embedding, X_out_valid_vader, X_in_valid_vader, X_valid_vader_leaves, \
        X_out_valid_liwc, X_in_valid_liwc, X_valid_liwc_leaves = X_valid
    X_out_test_embedding, X_in_test_embedding, X_out_test_vader, X_in_test_vader, X_test_vader_leaves, \
        X_out_test_liwc, X_in_test_liwc, X_test_liwc_leaves = X_test
    
    y_train, y_valid, y_test = y
    
    USE_SCHEDULER = True

    if MULTI_GPU:
        batch_size = batch_size * torch.cuda.device_count()
    num_workers = 32
    

    total_steps = math.ceil(len(X_out_train_embedding) / batch_size) * epochs
    warmup_steps = len(X_out_train_embedding) / batch_size

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

    stop_q = []

    for epoch in range(epochs):
        # Make data loader
        train_data_loader = create_data_loader((X_out_train_embedding, X_in_train_embedding,
                                                X_out_train_vader, X_in_train_vader,
                                                X_train_vader_leaves,
                                                X_out_train_liwc, X_in_train_liwc,
                                                X_train_liwc_leaves,
                                                y_train),
                                               batch_size, num_workers, data_collate_fn, True)
        valid_data_loader = create_data_loader((X_out_valid_embedding, X_in_valid_embedding,
                                                X_out_valid_vader, X_in_valid_vader,
                                                X_valid_vader_leaves,
                                                X_out_valid_liwc, X_in_valid_liwc,
                                                X_valid_liwc_leaves,
                                                y_valid),
                                               batch_size, num_workers, data_collate_fn, True)
        test_data_loader = create_data_loader((X_out_test_embedding, X_in_test_embedding,
                                               X_out_test_vader, X_in_test_vader,
                                               X_test_vader_leaves,
                                               X_out_test_liwc, X_in_test_liwc,
                                               X_test_liwc_leaves,
                                               y_test),
                                              batch_size, num_workers, data_collate_fn, True)

        train_loss = []
        valid_loss = []

        # Training Starts

        classifier.train()
        n = 0
        for d in train_data_loader:
            n += 1

            # Get a batch of X, F, y and load on GPU
            X_out_embedding, X_in_embedding, X_out_vader, X_in_vader, X_vader_leaves, X_out_liwc, X_in_liwc, X_liwc_leaves, y = d

            X_out_embedding = X_out_embedding.to(gpu)
            X_in_embedding = X_in_embedding.to(gpu)
            X_out_vader = X_out_vader.to(gpu)
            X_in_vader = X_in_vader.to(gpu)
            X_vader_leaves = X_vader_leaves.to(gpu)
            X_out_liwc = X_out_liwc.to(gpu)
            X_in_liwc = X_in_liwc.to(gpu)
            X_liwc_leaves = X_liwc_leaves.to(gpu)

            output, _ = classifier(X_out_embedding, X_in_embedding,
                                X_out_vader, X_in_vader,
                                X_vader_leaves,
                                X_out_liwc, X_in_liwc,
                                X_liwc_leaves)
            del X_out_embedding, X_in_embedding, X_out_vader, X_in_vader, X_vader_leaves, X_out_liwc, X_in_liwc, X_liwc_leaves

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
                X_out_embedding, X_in_embedding, X_out_vader, X_in_vader, X_vader_leaves, X_out_liwc, X_in_liwc, X_liwc_leaves, y = d

                X_out_embedding = X_out_embedding.to(gpu)
                X_in_embedding = X_in_embedding.to(gpu)
                X_out_vader = X_out_vader.to(gpu)
                X_in_vader = X_in_vader.to(gpu)
                X_vader_leaves = X_vader_leaves.to(gpu)
                X_out_liwc = X_out_liwc.to(gpu)
                X_in_liwc = X_in_liwc.to(gpu)
                X_liwc_leaves = X_liwc_leaves.to(gpu)

                output, _ = classifier(X_out_embedding, X_in_embedding,
                                    X_out_vader, X_in_vader,
                                    X_vader_leaves,
                                    X_out_liwc, X_in_liwc,
                                    X_liwc_leaves)
                del X_out_embedding, X_in_embedding, X_out_vader, X_in_vader, X_vader_leaves, X_out_liwc, X_in_liwc, X_liwc_leaves

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
            torch.save(classifier.state_dict(), 'luxemail_embedding.bin')
            min_valid_loss = avg_valid_loss

    print(str(run)+'th run, Minimum Valid Loss: ' + str(min_valid_loss))
    sys.stdout.flush()

    classifier.load_state_dict(torch.load('luxemail_embedding.bin'))

    classifier.eval()
    test_loss = []

    with torch.no_grad():
        predictions = []
        answers = []
        for d in test_data_loader:
            # Get a batch of X, F, y and load on GPU
            X_out_embedding, X_in_embedding, X_out_vader, X_in_vader, X_vader_leaves, X_out_liwc, X_in_liwc, X_liwc_leaves, y = d

            X_out_embedding = X_out_embedding.to(gpu)
            X_in_embedding = X_in_embedding.to(gpu)
            X_out_vader = X_out_vader.to(gpu)
            X_in_vader = X_in_vader.to(gpu)
            X_vader_leaves = X_vader_leaves.to(gpu)
            X_out_liwc = X_out_liwc.to(gpu)
            X_in_liwc = X_in_liwc.to(gpu)
            X_liwc_leaves = X_liwc_leaves.to(gpu)

            output, _ = classifier(X_out_embedding, X_in_embedding,
                                X_out_vader, X_in_vader,
                                X_vader_leaves,
                                X_out_liwc, X_in_liwc,
                                X_liwc_leaves)
            del X_out_embedding, X_in_embedding, X_out_vader, X_in_vader, X_vader_leaves, X_out_liwc, X_in_liwc, X_liwc_leaves

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

def save_embedding(classifier, X, y):
    X_train, X_valid, X_test = X
    X_out_train_embedding, X_in_train_embedding, X_out_train_vader, X_in_train_vader, X_train_vader_leaves, \
        X_out_train_liwc, X_in_train_liwc, X_train_liwc_leaves = X_train
    X_out_valid_embedding, X_in_valid_embedding, X_out_valid_vader, X_in_valid_vader, X_valid_vader_leaves, \
        X_out_valid_liwc, X_in_valid_liwc, X_valid_liwc_leaves = X_valid
    X_out_test_embedding, X_in_test_embedding, X_out_test_vader, X_in_test_vader, X_test_vader_leaves, \
        X_out_test_liwc, X_in_test_liwc, X_test_liwc_leaves = X_test
    
    y_train, y_valid, y_test = y
    
    train_data_loader = create_data_loader((X_out_train_embedding, X_in_train_embedding,
                                            X_out_train_vader, X_in_train_vader,
                                            X_train_vader_leaves,
                                            X_out_train_liwc, X_in_train_liwc,
                                            X_train_liwc_leaves,
                                            y_train),
                                           1, 1, data_collate_fn, False)
    valid_data_loader = create_data_loader((X_out_valid_embedding, X_in_valid_embedding,
                                            X_out_valid_vader, X_in_valid_vader,
                                            X_valid_vader_leaves,
                                            X_out_valid_liwc, X_in_valid_liwc,
                                            X_valid_liwc_leaves,
                                            y_valid),
                                           1, 1, data_collate_fn, False)
    test_data_loader = create_data_loader((X_out_test_embedding, X_in_test_embedding,
                                           X_out_test_vader, X_in_test_vader,
                                           X_test_vader_leaves,
                                           X_out_test_liwc, X_in_test_liwc,
                                           X_test_liwc_leaves,
                                           y_test),
                                          1, 1, data_collate_fn, False)
    
    with torch.no_grad():
        X_train_total_embedding = []
        for d in train_data_loader:
            X_out_embedding, X_in_embedding, X_out_vader, X_in_vader, X_vader_leaves, X_out_liwc, X_in_liwc, X_liwc_leaves, y = d

            X_out_embedding = X_out_embedding.to(gpu)
            X_in_embedding = X_in_embedding.to(gpu)
            X_out_vader = X_out_vader.to(gpu)
            X_in_vader = X_in_vader.to(gpu)
            X_vader_leaves = X_vader_leaves.to(gpu)
            X_out_liwc = X_out_liwc.to(gpu)
            X_in_liwc = X_in_liwc.to(gpu)
            X_liwc_leaves = X_liwc_leaves.to(gpu)

            _, embedding = classifier(X_out_embedding, X_in_embedding,
                                      X_out_vader, X_in_vader,
                                      X_vader_leaves,
                                      X_out_liwc, X_in_liwc,
                                      X_liwc_leaves)
            del X_out_embedding, X_in_embedding, X_out_vader, X_in_vader, X_vader_leaves, X_out_liwc, X_in_liwc, X_liwc_leaves
            embedding = embedding.squeeze(dim=0).to(cpu).tolist()
            X_train_total_embedding.append(embedding)
            
        X_valid_total_embedding = []
        for d in valid_data_loader:
            X_out_embedding, X_in_embedding, X_out_vader, X_in_vader, X_vader_leaves, X_out_liwc, X_in_liwc, X_liwc_leaves, y = d

            X_out_embedding = X_out_embedding.to(gpu)
            X_in_embedding = X_in_embedding.to(gpu)
            X_out_vader = X_out_vader.to(gpu)
            X_in_vader = X_in_vader.to(gpu)
            X_vader_leaves = X_vader_leaves.to(gpu)
            X_out_liwc = X_out_liwc.to(gpu)
            X_in_liwc = X_in_liwc.to(gpu)
            X_liwc_leaves = X_liwc_leaves.to(gpu)

            _, embedding = classifier(X_out_embedding, X_in_embedding,
                                      X_out_vader, X_in_vader,
                                      X_vader_leaves,
                                      X_out_liwc, X_in_liwc,
                                      X_liwc_leaves)
            del X_out_embedding, X_in_embedding, X_out_vader, X_in_vader, X_vader_leaves, X_out_liwc, X_in_liwc, X_liwc_leaves
            embedding = embedding.squeeze(dim=0).to(cpu).tolist()
            X_valid_total_embedding.append(embedding)

        X_test_total_embedding = []
        for d in test_data_loader:
            X_out_embedding, X_in_embedding, X_out_vader, X_in_vader, X_vader_leaves, X_out_liwc, X_in_liwc, X_liwc_leaves, y = d

            X_out_embedding = X_out_embedding.to(gpu)
            X_in_embedding = X_in_embedding.to(gpu)
            X_out_vader = X_out_vader.to(gpu)
            X_in_vader = X_in_vader.to(gpu)
            X_vader_leaves = X_vader_leaves.to(gpu)
            X_out_liwc = X_out_liwc.to(gpu)
            X_in_liwc = X_in_liwc.to(gpu)
            X_liwc_leaves = X_liwc_leaves.to(gpu)

            _, embedding = classifier(X_out_embedding, X_in_embedding,
                                      X_out_vader, X_in_vader,
                                      X_vader_leaves,
                                      X_out_liwc, X_in_liwc,
                                      X_liwc_leaves)
            del X_out_embedding, X_in_embedding, X_out_vader, X_in_vader, X_vader_leaves, X_out_liwc, X_in_liwc, X_liwc_leaves
            embedding = embedding.squeeze(dim=0).to(cpu).tolist()
            X_test_total_embedding.append(embedding)
    
    with open(DATA_DIR + 'X_train_total_embedding.pkl', 'wb') as f:
        pickle.dump(X_train_total_embedding, f)
    with open(DATA_DIR + 'X_valid_total_embedding.pkl', 'wb') as f:
        pickle.dump(X_valid_total_embedding, f)
    with open(DATA_DIR + 'X_test_total_embedding.pkl', 'wb') as f:
        pickle.dump(X_test_total_embedding, f)
        
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-run', '-n', help='Number of runs', default=10, dest='n_run')
    parser.add_argument('--random-seed', '-r', help='Start Seed', default=123, dest='seed')

    n_run = int(parser.parse_args().n_run)
    start_seed = int(parser.parse_args().seed)

    return n_run, start_seed

if __name__ == "__main__":
    n_run, start_seed = get_arguments()

    DATA_DIR = None # This should be private

    #OUTPUT_PATH = DATA_DIR + 'pair_profile_with_mixup_embedding.csv'

    MULTI_GPU = True
    MAX_LEN = 5
    MAX_EMAIL_NUM = 200
    BERT_HIDDEN = 768
    BERT_MAX_LEN = 512
    TEACHER = True
    BERT_TYPE = 'ELECTRA'

    # What to use
    TRANSFORMER = True
    VADER = True
    VADER_LEAVES = True
    LIWC = True
    LIWC_LEAVES = True

    if BERT_TYPE == 'BERT':
        from transformers import BertModel, BertTokenizer
        PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
        my_bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, return_dict=True)
    elif BERT_TYPE == 'ALBERT':
        from transformers import AlbertModel, AlbertTokenizer
        PRE_TRAINED_MODEL_NAME = 'albert-base-v2'
        tokenizer = AlbertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
        my_bert = AlbertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, return_dict=True)
    elif BERT_TYPE == 'ELECTRA':
        from transformers import ElectraModel, ElectraTokenizer
        PRE_TRAINED_MODEL_NAME = 'google/electra-base-discriminator'
        tokenizer = ElectraTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
        my_bert = ElectraModel.from_pretrained(PRE_TRAINED_MODEL_NAME, return_dict=True)

    UNSUP_FINETUNE_PATH = 'unsup_fine_tuning_' + BERT_TYPE + '.bin'
    
    if MULTI_GPU:
        gpu = torch.device('cuda')
        out_gpu = torch.device('cuda:2')
    else:
        gpu = torch.device('cuda:1')
    cpu = torch.device('cpu')

    loss_func = nn.CrossEntropyLoss(reduction='mean')
    loss_func = loss_func.to(out_gpu)
    
    X_out_train_embedding = pd.read_pickle(DATA_DIR + 'X_out_train_embedding.pkl')
    X_in_train_embedding = pd.read_pickle(DATA_DIR + 'X_in_train_embedding.pkl')
    X_out_valid_embedding = pd.read_pickle(DATA_DIR + 'X_out_valid_embedding.pkl')
    X_in_valid_embedding = pd.read_pickle(DATA_DIR + 'X_in_valid_embedding.pkl')
    X_out_test_embedding = pd.read_pickle(DATA_DIR + 'X_out_test_embedding.pkl')
    X_in_test_embedding = pd.read_pickle(DATA_DIR + 'X_in_test_embedding.pkl')

    P_train = pd.read_pickle(DATA_DIR + 'P_train.pkl')
    P_valid = pd.read_pickle(DATA_DIR + 'P_valid.pkl')
    P_test = pd.read_pickle(DATA_DIR + 'P_test.pkl')

    y_train = pd.read_pickle(DATA_DIR + 'y_train.pkl')
    y_valid = pd.read_pickle(DATA_DIR + 'y_valid.pkl')
    y_test = pd.read_pickle(DATA_DIR + 'y_test.pkl')    


    X_out_train_vader = pd.read_pickle(DATA_DIR + 'X_out_train_vader_aggregated.pkl')
    X_in_train_vader = pd.read_pickle(DATA_DIR + 'X_in_train_vader_aggregated.pkl')
    X_out_valid_vader = pd.read_pickle(DATA_DIR + 'X_out_valid_vader_aggregated.pkl')
    X_in_valid_vader = pd.read_pickle(DATA_DIR + 'X_in_valid_vader_aggregated.pkl')
    X_out_test_vader = pd.read_pickle(DATA_DIR + 'X_out_test_vader_aggregated.pkl')
    X_in_test_vader = pd.read_pickle(DATA_DIR + 'X_in_test_vader_aggregated.pkl')

    VADER_DIM, = X_out_train_vader[0].shape

    X_out_train_liwc = pd.read_pickle(DATA_DIR + 'X_out_train_liwc_aggregated.pkl')
    X_in_train_liwc = pd.read_pickle(DATA_DIR + 'X_in_train_liwc_aggregated.pkl')
    X_out_valid_liwc = pd.read_pickle(DATA_DIR + 'X_out_valid_liwc_aggregated.pkl')
    X_in_valid_liwc = pd.read_pickle(DATA_DIR + 'X_in_valid_liwc_aggregated.pkl')
    X_out_test_liwc = pd.read_pickle(DATA_DIR + 'X_out_test_liwc_aggregated.pkl')
    X_in_test_liwc = pd.read_pickle(DATA_DIR + 'X_in_test_liwc_aggregated.pkl')

    LIWC_DIM, = X_out_train_liwc[0].shape

    X_train_liwc_leaves = pd.read_pickle(DATA_DIR + 'X_train_liwc_xgboost_one_hot.pkl')
    X_valid_liwc_leaves = pd.read_pickle(DATA_DIR + 'X_valid_liwc_xgboost_one_hot.pkl')
    X_test_liwc_leaves = pd.read_pickle(DATA_DIR + 'X_test_liwc_xgboost_one_hot.pkl')

    LIWC_LEAVES_DIM_0, LIWC_LEAVES_DIM_1 = X_train_liwc_leaves[0].shape
    LIWC_LEAVES_DIM = LIWC_LEAVES_DIM_0 * LIWC_LEAVES_DIM_1

    X_train_vader_leaves = pd.read_pickle(DATA_DIR + 'X_train_vader_xgboost_one_hot.pkl')
    X_valid_vader_leaves = pd.read_pickle(DATA_DIR + 'X_valid_vader_xgboost_one_hot.pkl')
    X_test_vader_leaves = pd.read_pickle(DATA_DIR + 'X_test_vader_xgboost_one_hot.pkl')

    VADER_LEAVES_DIM_0, VADER_LEAVES_DIM_1 = X_train_vader_leaves[0].shape
    VADER_LEAVES_DIM = VADER_LEAVES_DIM_0 * VADER_LEAVES_DIM_1
    
    X_train = X_out_train_embedding, X_in_train_embedding, X_out_train_vader, X_in_train_vader, X_train_vader_leaves, \
        X_out_train_liwc, X_in_train_liwc, X_train_liwc_leaves
    X_valid = X_out_valid_embedding, X_in_valid_embedding, X_out_valid_vader, X_in_valid_vader, X_valid_vader_leaves, \
        X_out_valid_liwc, X_in_valid_liwc, X_valid_liwc_leaves
    X_test = X_out_test_embedding, X_in_test_embedding, X_out_test_vader, X_in_test_vader, X_test_vader_leaves, \
        X_out_test_liwc, X_in_test_liwc, X_test_liwc_leaves
    X = X_train, X_valid, X_test
    
    y = y_train, y_valid, y_test
    
    # Training parameters
    learning_rate = 1e-5
    #weight_decay = 0
    weight_decay = 1e-7
    batch_size = 4
    epochs = 20

    seed = start_seed
    dict_list = []
    
    max_auc = 0
    max_model = None
    
    for i in range(n_run):
        classifier = Classifier(
            rep_dim=2,
            num_layers=1,
            dropout=0.2,
            dim_features=256)

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
    
    save_embedding(max_model, X, y)
