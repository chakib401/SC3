from sklearn.kernel_approximation import RBFSampler, PolynomialCountSketch, Nystroem
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import kneighbors_graph
from time import time
from scipy import sparse
from time import time
import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
from utils import read_dataset, preprocess_dataset, clustering_accuracy
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import normalize
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import warnings
warnings.filterwarnings('ignore')

@tf.function
def convolve_rows(a, x, p):
  for _ in range(p):
    x = tf.sparse.sparse_dense_matmul(a, x)
  return x

@tf.function
def compute_loss(x, z, w):
  return tf.linalg.norm(x-z@(tf.transpose(z)@x@tf.transpose(w))@w)

@tf.function
def convolve_columns(a, x, p):
  for _ in range(p):
    x = x @ a
  return x

def ppmi(x):
  c = x.T @ x
  col_sums = c.sum(0)
  row_sums = c.sum(1)
  global_sum = c.sum()
  c = c * global_sum / col_sums / row_sums[:, None]
  c = np.log(c)
  c = np.nan_to_num(c)
  return np.maximum(c, 0)


def factorized_NJW(z, k):
  z_rsum = z.sum(0)
  zz_rsum = z @ z_rsum
  z_feat_map = z / zz_rsum[:,None]**.5
  z, _, _ = randomized_svd(z, k)
  z = KMeans(k).fit_predict(z)
  return z

def linear_feat_map(z, c=1):
  return np.column_stack([z, c*np.ones(len(z))])

def square_feat_map(z, c=1):
  polf = PolynomialFeatures(include_bias=True)
  x = polf.fit_transform(z)
  coefs = np.ones(len(polf.powers_))
  coefs[0] = c
  coefs[(polf.powers_ == 1).sum(1) == 2] = np.sqrt(2)
  coefs[(polf.powers_ == 1).sum(1) == 1] = np.sqrt(2*c) 
  return x * coefs


def approx_poly_feat_map(z, **kwargs):
  ps = PolynomialCountSketch(**kwargs)
  z = ps.fit_transform(z)
  return z


def apply_feature_map(feat_map, z, **kwargs):
  if feat_map == 'linear':
    return linear_feat_map(z, **kwargs)
  elif feat_map == 'squared': 
    return square_feat_map(z, **kwargs)
  elif feat_map == 'poly_approx': 
    return square_feat_map(z, **kwargs)
  elif feat_map == 'sampling': 
    sampling_feature = RBFSampler(**kwargs)
    return sampling_feature.fit_transform(z)
  elif feat_map == 'nystroem': 
    nystroem_feature = Nystroem(**kwargs)
    return nystroem_feature.fit_transform(z)

def convert_sparse_matrix_to_sparse_tensor(x):
    coo = x.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

# Parameters
flags.DEFINE_string('dataset', 'cora', 'Name of the graph dataset (acm, citeseer, pubmed or wiki).')
flags.DEFINE_string('kernel', 'linear', 'type of kernel')
dataset = flags.FLAGS.dataset
kernel = flags.FLAGS.kernel


metrics = {}
metrics['acc'] = []
metrics['nmi'] = []
metrics['ari'] = []
metrics['loss'] = []


adj, features, labels, n_classes = read_dataset(dataset)
tf_idf = dataset in ['cora', 'citeseer', 'acm']

norm_adj, features = preprocess_dataset(adj, features, tf_idf=tf_idf)
ppmi_adj, _ = preprocess_dataset(ppmi(features), [[0]])
ppmi_adj = ppmi_adj.toarray()

if dataset == 'acm': norm_adj = norm_adj.toarray()

x = features
x = x @ ppmi_adj

loss_old = np.inf

for power in range(1, 300):
  x = norm_adj @ x
    
  # main body
  z, _, w = randomized_svd(normalize(x, 'l1'), n_classes)
  loss = np.linalg.norm(x-z@(z.T@x@w.T)@w)

  z = apply_feature_map(kernel, z)
  z = factorized_NJW(z, n_classes)

  w = apply_feature_map(kernel, w.T)
  w = factorized_NJW(w, n_classes)


  metrics['loss'].append(loss)
  metrics['acc'].append(clustering_accuracy(labels, z)*100)
  metrics['nmi'].append(nmi(labels, z)*100)
  metrics['ari'].append(ari(labels, z)*100)
  
  if np.abs(loss_old-loss) < features.shape[1] / (features.shape[0]*np.ceil(np.sqrt(n_classes))): break
  loss_old = np.round(loss, 2)
  

results = {
    'mean': {k:(np.mean(v)).round(2) for k,v in metrics.items()}, 
    'std': {k:(np.std(v)).round(2) for k,v in metrics.items()}
}

print(f'Selected Power is {power}')

