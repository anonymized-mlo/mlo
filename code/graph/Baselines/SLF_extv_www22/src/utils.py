import argparse
import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from texttable import Texttable
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd


def parameter_parser():
    """
    Parse up command line parameters.
    """
    parser = argparse.ArgumentParser(description="Run SLF.")
    parser.add_argument("--edge-path",
                        nargs="?",
                        default="./input/WikiElec.txt",
                        help="Edge list in txt format.")
    parser.add_argument("--annotated-edge-path",
                        nargs="?",
                        default="../tr_edge_hr_att_083121.csv",
                        help="Edge list with split labels.")
    parser.add_argument("--outward-embedding-path",
                        nargs="?",
                        default="./output/outward",
                        help="Outward embedding path.")
    parser.add_argument("--inward-embedding-path",
                        nargs="?",
                        default="./output/inward",
                        help="Inward embedding path.")
    parser.add_argument("--epochs",
                        type=int,
                        default=20,
                        help="Number of training epochs. Default is 20.")
    parser.add_argument("--k1",
                        type=int,
                        default=32,
                        help="Dimension of positive SLF. Default is 32.")
    parser.add_argument("--k2",
                        type=int,
                        default=32,
                        help="Dimension of negative SLF. Default is 32.")
    parser.add_argument("--p0",
                        type=float,
                        default=0.001,
                        help="Effect of no feedback. Default is 0.001.")
    parser.add_argument("--n",
                        type=int,
                        default=5,
                        help="Number of noise samples. Default is 5.")
    parser.add_argument("--link-prediction",
                        type=bool,
                        default=False,
                        help="Make link prediction or not. Default is 5.")
    parser.add_argument("--sign-prediction",
                        type=bool,
                        default=True,
                        help="Make sign prediction or not. Default is 5.")
    parser.add_argument("--test-size",
                        type=float,
                        default=0.2,
                        help="Test ratio. Default is 0.2.")
    parser.add_argument("--split-seed",
                        type=int,
                        default=16,
                        help="Random seed for splitting dataset. Default is 16.")
    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.025,
                        help="Learning rate. Default is 0.025.")

    return parser.parse_args()


def fa(x, args):
    """
    Activation function f_a(x).
    """
    if x > 15:
        tmp = 1
    else:
        tmp = args.p0 * np.exp(x) / (1 + args.p0 * (np.exp(x) - 1))
    return tmp


def read_edge_list(args):
    """
    Load edges
    """
    G = nx.DiGraph()
    edges = pd.read_csv(args.annotated_edge_path).values

    for i in edges:
        G.add_edge(int(i[0]), int(i[1]), weight=i[2], label=i[3])
    edges = [[e[0], e[1], e[2]['weight'], e[2]['label']] for e in G.edges.data()]
    return edges, max(G.nodes) + 1  # index can start from 0.


@ignore_warnings(category=ConvergenceWarning)
def sign_prediction(out_emb, in_emb, train_edges, test_edges):
    """
    Evaluate the performance on the sign prediction task.
    :param out_emb: Outward embeddings.
    :param in_emb: Inward embeddings.
    :param train_edges: Edges for training the model.
    :param test_edges: Edges for test.
    """
    out_dim = out_emb.shape[1]
    in_dim = in_emb.shape[1]
    train_edges = train_edges
    train_x = np.zeros((len(train_edges), (out_dim + in_dim) * 2))
    train_y = np.zeros((len(train_edges), 1))

    for i, edge in enumerate(train_edges):
        u = edge[0]
        v = edge[1]
        if edge[2] > 0:
            train_y[i] = 1
        else:
            train_y[i] = 0
        train_x[i, : out_dim] = out_emb[u]
        train_x[i, out_dim: out_dim + in_dim] = in_emb[u]
        train_x[i, out_dim + in_dim: out_dim * 2 + in_dim] = out_emb[v]
        train_x[i, out_dim * 2 + in_dim:] = in_emb[v]

    test_edges = test_edges
    test_x = np.zeros((len(test_edges), (out_dim + in_dim) * 2))
    test_y = np.zeros((len(test_edges), 1))
    for i, edge in enumerate(test_edges):
        u = edge[0]
        v = edge[1]
        if edge[2] > 0:
            test_y[i] = 1
        else:
            test_y[i] = 0
        test_x[i, : out_dim] = out_emb[u]
        test_x[i, out_dim: out_dim + in_dim] = in_emb[u]
        test_x[i, out_dim + in_dim: out_dim * 2 + in_dim] = out_emb[v]
        test_x[i, out_dim * 2 + in_dim:] = in_emb[v]

    lr = LogisticRegression()
    lr.fit(train_x, train_y.ravel())
    test_y_score = lr.predict_proba(test_x)[:, 1]
    test_y_pred = lr.predict(test_x)
    auc_score = roc_auc_score(test_y, test_y_score)
    macro_f1_score = f1_score(test_y, test_y_pred, average='macro')
    pr, re, f1, _, = precision_recall_fscore_support(test_y, test_y_pred, average='macro')

    print('Sign Prediction')
    print("Precision: {:.4f}".format(pr), "Recall: {:.4f}".format(re), "F Score: {:.4f}".format(f1))

    return auc_score, macro_f1_score


def args_printer(args):
    """
    Print the parameters in tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    t = Texttable()
    l = [[k, args[k]] for k in args.keys()]
    l.insert(0, ["Parameter", "Value"])
    t.add_rows(l)
    print(t.draw())


def sign_prediction_printer(logs):
    """
    Print the performance on sign prediction task in tabular format.
    :param logs: Logs about the evaluation.
    """
    t = Texttable()
    epoch_list = logs['epoch']
    auc_list = logs['sign_prediction_auc']
    macrof1_list = logs['sign_prediction_macro_f1']
    l = [[epoch_list[i], auc_list[i], macrof1_list[i]] for i in range(len(epoch_list))]
    l.insert(0, ['Epoch', 'AUC', 'Macro-F1'])
    t.add_rows(l)
    print(t.draw())


def link_prediction_printer(logs):
    """
    Print the performance on link prediction task in tabular format.
    :param logs: Logs about the evaluation.
    """
    t = Texttable()
    epoch_list = logs['epoch']
    auc_p_list = logs['link_prediction_auc@p']
    auc_n_list = logs['link_prediction_auc@n']
    auc_non_list = logs['link_prediction_auc@non']
    l = [[epoch_list[i], auc_p_list[i], auc_n_list[i], auc_non_list[i]] for i in range(len(epoch_list))]
    l.insert(0, ['Epoch', 'AUC@p', 'AUC@n', 'AUC@non'])
    t.add_rows(l)
    print(t.draw())
