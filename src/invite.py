import networkx as nx
import numpy as np
import pandas as pd
import random
from functools import partial
import itertools
from tqdm.autonotebook import tqdm  # , trange

__all__ = [
    'graph_random_walk',
    'sample_censored',
]


def graph_random_walk(G, start=None, steps=None):
    """
    Sample a random walk on a networkx graph (G). By default, continues until
    all nodes in G are visited, unless  `steps` is set.

    Parameters
    ----------
    G : :obj: `networkx.Graph` or `networkx.DiGraph`
        A graph to sample the walk from
    start : int or str, optional
        Name of node to start walk from. Ignored if `node_prob` is set.
    steps : int
        Upper limit on number of transitions to record

    Returns
    -------
    `pandas.Series`
        A series of jumps, in order. Facilitates censoring.

    """

    N = nx.number_of_nodes(G)
    nodes = list(G)
    A = nx.to_numpy_array(G)     # adjacency matrix A

    # Right-stochastic transition matrix
    T = A/np.sum(A, axis=1)[:, np.newaxis]

    # pick the starting node

    p = np.zeros(N)
    if start is not None:           # provided start node?
        p[nodes.index(start)] = 1
    else:                           # no? --> random start node
        p[random.randint(0, N-1)] = 1

    visited = N*[False]      # check if all have been visited
    log = list()             # track which node was visited
    go_ahead = True

    node_idx = np.random.choice(range(N), p=p)
    visited[node_idx] = True     # been there, done that
    log.append(nodes[node_idx])  # got the t-shirt

    while (not all(visited)) and go_ahead:
        # need to transition away from current node
        p = T[node_idx, :]
        node_idx = np.random.choice(range(N), p=p)
        visited[node_idx] = True     # been there, done that
        log.append(nodes[node_idx])  # got the t-shirt
        if (steps is not None) and steps <= len(log):
            go_ahead = False
    return pd.Series(log)  # like an ordered dict. I like Pandas...


def sample_censored(G, n_walks, n_obsv,
                    demo=True, node_prob=None, safe=True,
                    **grw_kws):
    """

    Parameters
    ----------
    G
    n_walks
    n_obsv
    demo
    node_prob : list, array-like
        A list or array of node probabilities, ordered as `G.nodes`
    grw_kws

    Returns
    -------

    """

    steps = grw_kws.get('steps', None)
    start = grw_kws.get('start', None)

    if (node_prob is not None) and (start is not None):
        print('Node selection probability passed; ignoring `start`...')

    rw = partial(graph_random_walk, G, steps=steps)

    core_starts = np.random.choice(G.nodes(),
                                   size=len(G.nodes()),
                                   replace=False)
    m_core = [rw(start=i)[:2] for i in core_starts]
    # m_core = [[i] for i in G.nodes()]
    starts = np.random.choice(G.nodes, p=node_prob, size=n_walks)
    m = [rw(start=i).unique()[:n_obsv] for i in tqdm(starts)] + m_core

    missing_nodes = set(itertools.chain(*m)) != set(G.nodes())
    overtime = 0

    while missing_nodes and safe:
        print('SOMEHOW MISSING NODES!')
        overtime += 1
        start = np.random.choice(G.nodes, p=node_prob)
        m += [rw(start=start).unique()[:n_obsv]]
        missing_nodes = set(itertools.chain(*m)) != set(G.nodes())
    if overtime:
        print(f'Sampled {overtime} extra walks to ensure node coverage!')
    if demo:
        for n, i in enumerate(m[:4]):
            print(n, ' → '.join('{:^2}'.format(
                G.nodes[j]['item']) for j in i.tolist())
            )
            print('\n'.join(i for i in 2 * ['.']))
        print(len(m) - 1, ' → '.join('{:^2}'.format(
            G.nodes[j]['item']) for j in m[-1].tolist())
        )

    return m
