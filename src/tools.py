from sklearn import preprocessing as pre
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pomegranate as pom
import pandas as pd
from itertools import product
import networkx as nx


def minmax(x):
    x[x < 0] = 0
    return (x-np.min(x))/np.ptp(x)


def softmax(x, axis=1):
    e = np.exp(x)
    return e/e.sum(axis=axis)[:, None]


def fill_diagonal(source_array, diagonal):
    copy = source_array.copy()
    np.fill_diagonal(copy, diagonal)
    return copy


def labeled_adj(names, X):
    return pd.DataFrame(X, columns=names, index=names)


def symmetrize(x):
    bott = np.tril(x) + np.tril(x).T
    top = np.triu(x) + np.triu(x).T
    # return (bott+top)/2. + infs
    return np.maximum(bott, top)


def cosine_model(m, classes=None, **cos_kws):

    encode = pre.MultiLabelBinarizer(classes=classes)
    tags = encode.fit_transform(m)

    cos = cosine_similarity(tags.T.dot(tags))
    cos -= np.diag(cos.diagonal())
    return cos


def markov_model(m, classes=None, **pom_kws):
    # if classes is not None:

    clf = pom.MarkovChain.from_samples(np.array(m).tolist(), **pom_kws)
    # states = clf.distributions[0].keys()
    # weights = [(i, j, np.exp(max(
    #     clf.log_probability([i,j]),
    #     clf.log_probability([j,i])
    # ))) for i, j in product(states, states)]
    C = pd.DataFrame(
        clf.distributions[1].parameters[0],
        # weights,
        columns=['source', 'target', 'weight']
    ).fillna(0)
    C_g = nx.from_pandas_edgelist(C, edge_attr=True, create_using=nx.DiGraph)
    mkv = nx.to_numpy_array(C_g)
    mkv -= np.diag(mkv.diagonal())
    return mkv


def hidden_markov(m, n_components, n_classes, **pom_kws):
    clf = pom.HiddenMarkovModel.from_samples(
        pom.distributions.DiscreteDistribution,
        n_components, np.array(m).tolist()
    )
    # states = clf.distributions[0].keys()
    states = range(n_classes)
    weights = [(i, j, np.exp(max(
        clf.log_probability([i, j]),
        clf.log_probability([j, i])
    ))) for i, j in product(states, states)]
    C = pd.DataFrame(
        # clf.distributions[1].parameters[0],
        weights,
        columns=['source', 'target', 'weight']
    ).fillna(0)
    C_g = nx.from_pandas_edgelist(C, edge_attr=True, create_using=nx.DiGraph)
    mkv = nx.to_numpy_array(C_g)
    mkv -= np.diag(mkv.diagonal())
    return mkv
    # return clf.dense_transition_matrix()


def local_and_global_consistency(G, alpha=0.99,
                                 max_iter=30,
                                 label_name='label',
                                 return_prob=True):
    """MODIFIED TO RETURN PROBABILITIES ON F
    Node classification by Local and Global Consistency

    References
    ----------
    Zhou, D., Bousquet, O., Lal, T. N., Weston, J., & Schölkopf, B. (2004).
    Learning with local and global consistency.
    Advances in neural information processing systems, 16(16), 321-328.
    """
    from networkx.algorithms.node_classification.utils import (
        _get_label_info,
        _init_label_matrix,
        _propagate,
        _predict,
    )
    import numpy as np
    from scipy import sparse

    def _build_propagation_matrix(X, labels, alpha):
        degrees = X.sum(axis=0).A[0]
        degrees[degrees == 0] = 1  # Avoid division by 0
        D2 = np.sqrt(sparse.diags((1.0 / degrees), offsets=0))
        S = alpha * D2.dot(X).dot(D2)
        return S

    def _build_base_matrix(X, labels, alpha, n_classes):
        n_samples = X.shape[0]
        B = np.zeros((n_samples, n_classes))
        B[labels[:, 0], labels[:, 1]] = 1 - alpha
        return B

    X = nx.to_scipy_sparse_matrix(G)  # adjacency matrix
    labels, label_dict = _get_label_info(G, label_name)

    if labels.shape[0] == 0:
        raise nx.NetworkXError(
            "No node on the input graph is labeled by '" + label_name + "'.")

    n_samples = X.shape[0]
    n_classes = label_dict.shape[0]
    F = _init_label_matrix(n_samples, n_classes)

    P = _build_propagation_matrix(X, labels, alpha)
    B = _build_base_matrix(X, labels, alpha, n_classes)

    remaining_iter = max_iter
    while remaining_iter > 0:
        F = _propagate(P, F, B)
        remaining_iter -= 1

    predicted = _predict(F, label_dict)
    if return_prob:
        return (predicted, pd.DataFrame({
            label_dict[i]: F[:, i] for i in range(F.shape[1])
        }, index=list(G.nodes())))
    return predicted
