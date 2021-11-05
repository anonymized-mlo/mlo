"""Data reading utils."""

import json
import numpy as np
import pandas as pd
from scipy import sparse
from texttable import Texttable
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle


def read_graph(args):
    """
    Method to read graph and create a target matrix with pooled adjacency matrix powers.
    :param args: Arguments object.
    :return edges: Edges dictionary.
    """
    dataset = pd.read_csv(args.edge_path).values.tolist()
    edges = {}
    edges["positive_edges"] = {}
    edges["negative_edges"] = {}

    edges["positive_edges"]['train'] = [edge[0:2] for edge in dataset if (edge[2] == 1 and edge[3] == 'train')]
    edges["positive_edges"]['test'] = [edge[0:2] for edge in dataset if (edge[2] == 1 and edge[3] == 'test')]
    edges["negative_edges"]['train'] = [edge[0:2] for edge in dataset if (edge[2] == -1 and edge[3] == 'train')]
    edges["negative_edges"]['test'] = [edge[0:2] for edge in dataset if (edge[2] == -1 and edge[3] == 'test')]

    edges["ecount"] = len(dataset)
    edges["ncount"] = len(set([edge[0] for edge in dataset]+[edge[1] for edge in dataset]))
    return edges

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])

def calculate_auc(targets, predictions, edges):
    """
    Calculate performance measures on test dataset.
    :param targets: Target vector to predict.
    :param predictions: Predictions vector.
    :param edges: Edges dictionary with number of edges etc.
    :return auc: AUC value.
    :return f1: F1-score.
    """
    neg_ratio = len(edges["negative_edges"])/edges["ecount"]
    targets = [0 if target == 1 else 1 for target in targets]

    with open('targets.pkl', 'wb') as h:
        pickle.dump(targets, h)
    with open('predictions.pkl', 'wb') as k:
        pickle.dump(predictions, k)

    auc = roc_auc_score(targets, predictions)
    f1 = f1_score(targets, np.round(predictions), average='macro')
    precision, recall, _, _ = precision_recall_fscore_support(targets, np.round(predictions), average='macro')
    accuracy = accuracy_score(targets, np.round(predictions))

    return auc, f1, predictions, precision, recall, accuracy

def score_printer(logs, avg=None):
    """
    Print the performance for every 10th epoch on the test dataset.
    :param logs: Log dictionary.
    """

    if (avg is not None):
        return logs["performance"][-1][1],  logs["performance"][-1][2],  logs["performance"][-1][3],  logs["performance"][-1][4], logs["performance"][-1][5]

def save_logs(args, logs):
    """
    Save the logs at the path.
    :param args: Arguments objects.
    :param logs: Log dictionary.
    """
    with open(args.log_path, "w") as f:
        json.dump(logs, f)

def setup_features(args, positive_edges, negative_edges, node_count):
    """
    Setting up the node features as a numpy array.
    :param args: Arguments object.
    :param positive_edges: Positive edges list.
    :param negative_edges: Negative edges list.
    :param node_count: Number of nodes.
    :return X: Node features.
    """
    if args.spectral_features:
        X = create_spectral_features(args, positive_edges, negative_edges, node_count)
    else:
        X = create_general_features(args)
    return X

def create_general_features(args):
    """
    Reading features using the path.
    :param args: Arguments object.
    :return X: Node features.
    """
    X = np.array(pd.read_csv(args.features_path))
    cols = pd.read_csv(args.features_path).columns

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X = np.array(pd.DataFrame(data=X, columns = cols))

    return X

def create_spectral_features(args, positive_edges, negative_edges, node_count):
    """
    Creating spectral node features using the train dataset edges.
    :param args: Arguments object.
    :param positive_edges: Positive edges list.
    :param negative_edges: Negative edges list.
    :param node_count: Number of nodes.
    :return X: Node features.
    """
    p_edges = positive_edges + [[edge[1], edge[0]] for edge in positive_edges]
    n_edges = negative_edges + [[edge[1], edge[0]] for edge in negative_edges]
    train_edges = p_edges + n_edges
    index_1 = [edge[0] for edge in train_edges]
    index_2 = [edge[1] for edge in train_edges]
    values = [1]*len(p_edges) + [-1]*len(n_edges)
    shaping = (node_count, node_count)
    signed_A = sparse.csr_matrix(sparse.coo_matrix((values, (index_1, index_2)),
                                                   shape=shaping,
                                                   dtype=np.float32))

    svd = TruncatedSVD(n_components=args.reduction_dimensions,
                       n_iter=args.reduction_iterations,
                       random_state=args.seed)
    svd.fit(signed_A)
    X = svd.components_.T
    return X
