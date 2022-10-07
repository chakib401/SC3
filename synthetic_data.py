def _shuffle(data, random_state=None):
    generator = check_random_state(random_state)
    n_rows, n_cols = data.shape
    row_idx = generator.permutation(n_rows)
    col_idx = generator.permutation(n_cols)
    result = data[row_idx][:, col_idx]
    return result, row_idx, col_idx


import numbers
import array
import warnings
from collections.abc import Iterable

import numpy as np
from scipy import linalg
import scipy.sparse as sp

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import check_array, check_random_state
from sklearn.utils import shuffle as util_shuffle
from sklearn.utils.random import sample_without_replacement

def make_biclusters(
    shape,
    n_clusters,
    a=None, 
    b=None,
    *,
    noise=0.0,
    minval=10,
    maxval=100,
    shuffle=True,
    random_state=None,
):
    """Generate a constant block diagonal structure array for biclustering.
    Read more in the :ref:`User Guide <sample_generators>`.
    Parameters
    ----------
    shape : iterable of shape (n_rows, n_cols)
        The shape of the result.
    n_clusters : int
        The number of biclusters.
    noise : float, default=0.0
        The standard deviation of the gaussian noise.
    minval : int, default=10
        Minimum value of a bicluster.
    maxval : int, default=100
        Maximum value of a bicluster.
    shuffle : bool, default=True
        Shuffle the samples.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    Returns
    -------
    X : ndarray of shape `shape`
        The generated array.
    rows : ndarray of shape (n_clusters, X.shape[0])
        The indicators for cluster membership of each row.
    cols : ndarray of shape (n_clusters, X.shape[1])
        The indicators for cluster membership of each column.
    See Also
    --------
    make_checkerboard: Generate an array with block checkerboard structure for
        biclustering.
    References
    ----------
    .. [1] Dhillon, I. S. (2001, August). Co-clustering documents and
        words using bipartite spectral graph partitioning. In Proceedings
        of the seventh ACM SIGKDD international conference on Knowledge
        discovery and data mining (pp. 269-274). ACM.
    """
    generator = check_random_state(random_state)
    n_rows, n_cols = shape
    consts = generator.uniform(minval, maxval, n_clusters)

    # row and column clusters of approximately equal sizes
    if a is None and b is None:
      a = np.repeat(1.0 / n_clusters, n_clusters)
      b = np.repeat(1.0 / n_clusters, n_clusters)
    row_sizes = generator.multinomial(n_rows, a)
    col_sizes = generator.multinomial(n_cols, b)

    row_labels = np.hstack(
        list(np.repeat(val, rep) for val, rep in zip(range(n_clusters), row_sizes))
    )
    col_labels = np.hstack(
        list(np.repeat(val, rep) for val, rep in zip(range(n_clusters), col_sizes))
    )

    result = np.zeros(shape, dtype=np.float64)
    for i in range(n_clusters):
        selector = np.outer(row_labels == i, col_labels == i)
        result[selector] += consts[i]

    if noise > 0:
        result += generator.normal(scale=noise, size=result.shape)

    if shuffle:
        result, row_idx, col_idx = _shuffle(result, random_state)
        row_labels = row_labels[row_idx]
        col_labels = col_labels[col_idx]

    rows = np.vstack([row_labels == c for c in range(n_clusters)])
    cols = np.vstack([col_labels == c for c in range(n_clusters)])

    return result, rows, cols


def make_checkerboard(
    shape,
    n_clusters,
    a=None,
    b=None,
    *,
    noise=0.0,
    minval=10,
    maxval=100,
    shuffle=True,
    random_state=None,
):
    """Generate an array with block checkerboard structure for biclustering.
    Read more in the :ref:`User Guide <sample_generators>`.
    Parameters
    ----------
    shape : tuple of shape (n_rows, n_cols)
        The shape of the result.
    n_clusters : int or array-like or shape (n_row_clusters, n_column_clusters)
        The number of row and column clusters.
    noise : float, default=0.0
        The standard deviation of the gaussian noise.
    minval : int, default=10
        Minimum value of a bicluster.
    maxval : int, default=100
        Maximum value of a bicluster.
    shuffle : bool, default=True
        Shuffle the samples.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    Returns
    -------
    X : ndarray of shape `shape`
        The generated array.
    rows : ndarray of shape (n_clusters, X.shape[0])
        The indicators for cluster membership of each row.
    cols : ndarray of shape (n_clusters, X.shape[1])
        The indicators for cluster membership of each column.
    See Also
    --------
    make_biclusters : Generate an array with constant block diagonal structure
        for biclustering.
    References
    ----------
    .. [1] Kluger, Y., Basri, R., Chang, J. T., & Gerstein, M. (2003).
        Spectral biclustering of microarray data: coclustering genes
        and conditions. Genome research, 13(4), 703-716.
    """
    generator = check_random_state(random_state)

    if hasattr(n_clusters, "__len__"):
        n_row_clusters, n_col_clusters = n_clusters
    else:
        n_row_clusters = n_col_clusters = n_clusters

    # row and column clusters of approximately equal sizes
    n_rows, n_cols = shape
    if a is None and b is None:
      a = np.repeat(1.0 / n_clusters, n_clusters)
      b = np.repeat(1.0 / n_clusters, n_clusters)
    row_sizes = generator.multinomial(n_rows, a)
    col_sizes = generator.multinomial(n_cols, b)

    row_labels = np.hstack(
        list(np.repeat(val, rep) for val, rep in zip(range(n_row_clusters), row_sizes))
    )
    col_labels = np.hstack(
        list(np.repeat(val, rep) for val, rep in zip(range(n_col_clusters), col_sizes))
    )

    result = np.zeros(shape, dtype=np.float64)
    for i in range(n_row_clusters):
        for j in range(n_col_clusters):
            selector = np.outer(row_labels == i, col_labels == j)
            result[selector] += generator.uniform(minval, maxval)

    if noise > 0:
        result += generator.normal(scale=noise, size=result.shape)

    if shuffle:
        result, row_idx, col_idx = _shuffle(result, random_state)
        row_labels = row_labels[row_idx]
        col_labels = col_labels[col_idx]

    rows = np.vstack(
        [
            row_labels == label
            for label in range(n_row_clusters)
            for _ in range(n_col_clusters)
        ]
    )
    cols = np.vstack(
        [
            col_labels == label
            for _ in range(n_row_clusters)
            for label in range(n_col_clusters)
        ]
    )

    return result, rows, cols


import scipy.sparse as sp
from gcc.utils import read_dataset, preprocess_dataset
from sklearn.metrics import silhouette_score, davies_bouldin_score

def A():
  n_classes = 10
  features, rows, cols = make_biclusters((500, 500), n_classes, 
                                        noise=30.0, minval=10, maxval=100, 
                                        shuffle=True, 
                                        random_state=0)
  return features, rows, cols, n_classes, 'A', None, None

def B():
  n_classes = 6
  r=[2,25,3, 5, 10, 1];r=np.array(r)/sum(r)
  c=[2,10,33,32, 6, 10];c=np.array(c)/sum(c)
  features, rows, cols = make_biclusters((800, 1000), n_classes, 
                                        noise=40.0, minval=10, maxval=100, 
                                        shuffle=True, 
                                        random_state=0,
                                        a=r,b=c)
  return features, rows, cols, n_classes, 'B', r, c

def C():
  n_classes = 8
  features, rows, cols = make_checkerboard((800, 800), n_classes, 
                                        noise=40.0, minval=10, maxval=100, 
                                        shuffle=True, 
                                        random_state=0)
  return features, rows, cols, n_classes, 'C', None, None

def D():
  n_classes = 7
  r=[1,.05,.1,.2,1,.5,1];r=np.array(r)/sum(r)
  c=[.2,.1,.25,1,1,.5,1];c=np.array(c)/sum(c)
  features, rows, cols = make_checkerboard((2000, 1200), n_classes, 
                                        noise=40.0, minval=10, maxval=100, 
                                        shuffle=True, 
                                        random_state=0,
                                        a=r,b=c)
  return features, rows, cols, n_classes, 'D', r, c


def E():
  n_classes = 15
  r=[ 7,  1,  2,  8,  5, 19,  4,  7, 18,  9,  5, 17, 18,  9, 16];r=np.array(r)/sum(r)
  c=[ 6, 19, 18, 12, 12, 10, 13, 19, 14,  3, 10,  7, 16, 15, 10];c=np.array(c)/sum(c)
  features, rows, cols = make_biclusters((2500, 1500), n_classes, 
                                        noise=40.0, minval=10, maxval=100, 
                                        shuffle=True, 
                                        random_state=0,
                                        a=r,b=c)
  return features, rows, cols, n_classes, 'E', r, c

%cd /content/gcc
from sklearn.utils.extmath import randomized_svd

from time import time
import tensorflow as tf
import numpy as np
from gcc.metrics import output_metrics, print_metrics
from gcc.utils import read_dataset, preprocess_dataset
from sklearn.kernel_approximation import RBFSampler, PolynomialCountSketch, Nystroem
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import kneighbors_graph
from time import time
from scipy import sparse
from sklearn.metrics import silhouette_score
import tensorflow as tf

def pmi(df, positive=True):
  col_totals = df.sum(axis=0)
  total = col_totals.sum()
  row_totals = df.sum(axis=1)
  expected = np.outer(row_totals, col_totals) / total
  df = df / expected
  # Silence distracting warnings about log(0):
  with np.errstate(divide='ignore'):
    df = np.log(df)
  df[~np.isfinite(df)] = 0.0  # log(0) = 0
  if positive:
    df[df < 0] = 0.0
  return df  


def average_pmi_per_cluster(x, labels):
  values = 0
  pmi_mat = pmi(x @ x.T, positive=False)
  for c in np.unique(labels):
    intra = pmi_mat[labels == c][:, labels == c]
    inter = pmi_mat[labels == c][:, labels != c]
    v = np.mean(intra) - np.mean(inter)
    values += v * np.sum(labels == c) / len(labels)
  return values

@tf.function
def convolve(feature, adj_normalized, power):
  for _ in range(power):
    feature = tf.sparse.sparse_dense_matmul(adj_normalized, feature)
  return feature

n_runs = 5



n_runs = 10
n_clusters = 0

p = 10

for dataset in [A(), B(), C(), D(), E()]:

  
  n_runs = 10

  features, rows, cols,n_classes, name, r, c = dataset

  adj = kneighbors_graph(features, 3, mode='connectivity')

  rows, cols = rows.argmax(0), cols.argmax(0)
  
  
  
  
  n, d = features.shape
  k = n_classes

  metrics = {}
  metrics['time'] = []
  metrics['acc'] = []
  metrics['nmi'] = []
  metrics['ari'] = []
  metrics['cos'] = []
    
  # print(dataset, '-----------')
  
  tf_idf = True
  n, d = features.shape

  norm_adj, features = preprocess_dataset(adj, features, 
                                          tf_idf=tf_idf,
                                          sparse=False)
  
  ppmi_mat = ppmi(features)
  y_indices = np.argsort(ppmi_mat)[:, :top_n].reshape(-1)
  x_indices = np.repeat(np.arange(d), top_n)
  
  values =  np.sort(ppmi_mat)[:, :top_n].reshape(-1)
  ppmi_mat = sparse.csr_matrix((values, (x_indices, y_indices)), shape=(d,d))
  col_norm_adj, _ = preprocess_dataset(ppmi_mat, features, 
                                          tf_idf=tf_idf,
                                          sparse=False)
  
  
  n, d = features.shape
  k = n_classes
    
    
  metrics = {}
  metrics['pmi'] = []
  metrics['acc'] = []
  metrics['nmi'] = []
  metrics['ari'] = []
  metrics['time'] = []
  
  def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

  norm_adj = convert_sparse_matrix_to_sparse_tensor(norm_adj.astype('float64'))
  features = tf.convert_to_tensor(features.astype('float64'))
  
  x = features
  for run in range(n_runs):
    features = x
    t0 = time()
    features = features @ col_norm_adj
      
    
    features = convolve(features, norm_adj, p)
    
    P, _, Q, _ = run_model(features, c=2**-.5, k=k, coclustering=True)
    
    metrics['time'].append(time()-t0)
    a = clustering_accuracy(rows, P)
    b = clustering_accuracy(cols, Q)
    metrics['acc'].append((a+b-a*b)*100)

  results = {
      'mean': {k:(np.mean(v)).round(2) for k,v in metrics.items() }, 
      'std': {k:(np.std(v)).round(2) for k,v in metrics.items()}
  }

  means = results['mean']
  std = results['std']


  print(f"{name} {p}")
  print(f"{means['acc']}±{std['acc']}", sep=',')


