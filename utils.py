from sklearn import metrics
from sklearn.metrics import confusion_matrix, silhouette_score, davies_bouldin_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer
import os.path
import scipy.io as sio
import scipy.sparse as sp


def ordered_confusion_matrix(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred)
    w = np.max(conf_mat) - conf_mat
    row_ind, col_ind = linear_sum_assignment(w)
    conf_mat = conf_mat[row_ind, :]
    conf_mat = conf_mat[:, col_ind]
    return conf_mat


def clustering_accuracy(y_true, y_pred):
    conf_mat = ordered_confusion_matrix(y_true, y_pred)
    return np.trace(conf_mat) / np.sum(conf_mat)


def aug_normalized_adjacency(adj, add_loops=True):
    if add_loops:
        adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def row_normalize(mx, add_loops=True):
    if add_loops:
        mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def read_dataset(dataset, sparse_adj=True):
    data = sio.loadmat(os.path.join('data', f'{dataset}.mat'))
    features = data['fea'].astype(float)
    adj = data['W']
    adj = adj.astype(float)
    if sparse_adj:
        adj = sp.csc_matrix(adj)
    if sp.issparse(features):
        features = features.toarray()
    labels = data['gnd'].reshape(-1) - 1
    n_classes = len(np.unique(labels))
    return adj, features, labels, n_classes


def preprocess_dataset(adj, features, row_norm=True, sym_norm=True, feat_norm=True, tf_idf=False):
    if sym_norm:
        adj = aug_normalized_adjacency(adj, True)
    if row_norm:
        adj = row_normalize(adj, True)

    if tf_idf:
        features = TfidfTransformer().fit_transform(features).toarray()
    if feat_norm:
        features = normalize(features)
    return adj, features

