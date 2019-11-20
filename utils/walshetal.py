import sys
import os
from pathlib import Path
from functools import partial
from itertools import cycle
import argparse
import logging
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
from textwrap import fill

import utils.plot_utils as pu
from src.models import CensoredRW, train
from src.analysis import draw_G, test_multi, compare_mats, compare_pvr
from src.invite import sample_censored

from src.tools import cosine_model, markov_model  # , hidden_markov
from src.tools import minmax, symmetrize, fill_diagonal, labeled_adj

sys.path.insert(0, os.path.abspath('..'))


toplvl_directory = Path(__file__).parents[1]
net_path = toplvl_directory/'data'/'network_models_shared'
latex_path = toplvl_directory/'latex'/'jmd19'/'img'/'walshetal'
latex_path.mkdir(parents=True, exist_ok=True)
log_path = latex_path/'logfile-walshetal.txt'


def get_struct(model):
    struct = pd.read_excel(
        net_path/f'{model}_component_model.xlsx', index_col=0)
    if model == 'aircraft':
        edgel = struct.stack().reset_index()
        edgel = edgel[edgel[0] != 0]
        edgel['level_0'] = (
            edgel.level_0
            .str.split(' ')
            .str[:-1]
            .str.join(' ')
        )
        edgel['level_1'] = (
            edgel.level_1
            .str.split(' ')
            .str[:-1]
            .str.join(' ')
        )
        struct = (
            edgel
            .groupby(['level_0', 'level_1'])
            .sum().unstack()[0]
            .fillna(0)
        )
        struct[struct > 1.] = 1
    return struct


def plot_ivt(G, pos, MWOs, title, txtpos=.5, colwidth=50, withlabels=True):

    palette = sns.color_palette("muted", n_colors=len(MWOs)).as_hex()

    plt.figure(figsize=pu.figsize(columns=2))

    mwotxts = [f'#{n+1} ' + fill(
        f"{[G.node[i]['item'] for i in a]}",
        width=colwidth, subsequent_indent='      '
    ) for n, a in enumerate(MWOs)]

    plt.text(txtpos, .75, 'Tagged MWOs',
             fontsize=pu.font_size())

    rads = cycle([0.15, 0.2, 0.25, .3])
    for n, a in enumerate(MWOs):
        edgelist = [(a[n], a[n+1]) for n in range(len(a)-1)]
        jump_edges = nx.draw_networkx_edges(
            nx.DiGraph(G),
            pos=pos, edgelist=edgelist,
            width=2, alpha=.8,
            edge_color=palette[n],
            label=f'MWO: {n}',
            connectionstyle=f'arc3,rad={next(rads)}',
        )
        [i.set_zorder(0) for i in jump_edges]

        plt.text(txtpos, .6-.15*n-.15*(len(MWOs)-3),  mwotxts[n],
                 color=palette[n], fontsize=pu.font_size())
    draw_G(G, pos, node_size=20, withlabels=withlabels,
           font_size=pu.label_size(), font_family='serif',
           title=title)
    plt.tight_layout()


def thres_graph(G, pos, A_thres, name, withlabels=True):
    A = nx.to_numpy_array(G)

    f = plt.figure(figsize=pu.figsize(aspect=1.618))
    in_approx = nx.from_numpy_array(A_thres - A > 0, create_using=nx.Graph)
    not_in_approx = nx.from_numpy_array(A - A_thres > 0, create_using=nx.Graph)
#         print((d['thres'] - A > 0).sum(), (A - d['thres'] > 0).sum())
    draw_G(G, pos, fp=in_approx, fn=not_in_approx,
           node_size=20, withlabels=withlabels,
           font_size=pu.ticks_size(), font_family='serif',
           legend=False)
    plt.title(name)
    return f


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate plot/results for Supporting Walsh et al Nets'
    )
    parser.add_argument('-m', '--model', default='drivetrain',
                        type=str, choices=['aircraft',
                                           'drivetrain', 'bicycle'],
                        help='which of the Walsh physical networks to test')
    parser.add_argument('-n', '--nwalks', default=20, type=int,
                        help='number of synthetic mwos to INVITE sample')
    parser.add_argument('-l', '--length', default=4, type=int,
                        help='length of synthetic mwos via INVITE sample')
    parser.add_argument('-r', '--seed', default=8, type=int,
                        help='(r)andom seed for numpy, for reproducibility')
    parser.add_argument('-s', '--save', action='store_true', default=False)
    args = parser.parse_args()

    pu.figure_setup()
    ps = partial(pu.plot_or_save, latex_path, args,
                 transparent=True)
    np.random.seed(args.seed)

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%d/%m/%Y %H:%M:%S',
    )
    logging.info('############ NEW RUN #############')
    logging.info("Run Parameters:\n" + str(args))

    # ########## GET DATA ########### #

    model = args.model
    struct = get_struct(model)
    # plt.spy(struct)
    logging.info('\n'.join([
        'Shape and Sparsity: ',
        f'{struct.shape}',
        f'{1 - struct.sum().sum() / struct.shape[0] ** 2:.1%}'
    ]))

    G = nx.from_pandas_adjacency(struct)  # build a graph
    G = nx.convert_node_labels_to_integers(G, label_attribute='item')
    logging.info(f'Clustering: C_beta = {nx.average_clustering(G):.2f}')

    pos = nx.layout.kamada_kawai_layout(G)  # plotting layout
    A = nx.to_numpy_array(G)  # numpy adj. matrix representation

    # ########## INVITE Samples ########### #
    M = sample_censored(G, args.nwalks, args.length, steps=30, demo=False)
    logging.info(f'Sampled {args.nwalks} walks of length {args.length}\n'
                 f'Required {len(M)} to cover all nodes.')

    lab = False if model == 'aircraft' else True
    if model == 'drivetrain':  # looks way better flipped...
        pos = {k: np.array([-1, 1])*v for k, v in pos.items()}
        plot_ivt(G, pos, M[:4], model, txtpos=.3, withlabels=lab)

    else:
        plot_ivt(G, pos, M[:4], model, txtpos=.5, withlabels=lab)

    ps(f'inviteRW')

    N = struct.shape[0]
    cuda = False  # if N > 50 else True
    approx = CensoredRW(N, sym=True, cuda=cuda)

    learning_rate = 0.1

    train(approx, np.array(M), compare_true=A, batch_size=min(500, len(M)),
          epochs=50, callback=True, lr=learning_rate)

    res = fill_diagonal(approx.P.detach().cpu().numpy(), 0)

    def prep(A): return labeled_adj(struct.columns, minmax(symmetrize(A)))

    mkv1 = markov_model(M, k=1)
    mkv2 = markov_model(M, k=2)
    # mkv2 = hidden_markov(M, max(map(len, M)), N, n_jobs=4)
    cos = cosine_model(M)

    models = dict(
        INVITE=prep(res),
        Cosine=prep(cos),
        MC1=prep(mkv1),
        MC2=prep(mkv2),
    )

    log = test_multi(A, **models)

    compare_pvr(A, log)
    ps(f'PvR')

    fig = plt.figure(figsize=pu.figsize(columns=2))
    labels = struct.columns if models == 'bicycle' else None
    compare_mats(
        A,
        labels=labels,
        **models
    )
    ps(f'matrix_compare', fig)

    custom_cycler = (cycler(color=sns.color_palette(n_colors=len(log)))
                     + cycler(ls=['-', ':', '--', '-.']))

    f, ax = plt.subplots(figsize=pu.figsize(aspect=1.6))
    ax.set_prop_cycle(custom_cycler)

    for name, d in log.items():
        ax.plot(d['t'], d['f'], label=name)

        # ax.legend(loc=0, fontsize=pu.label_size())
    # plt.legend(loc='upper center',
    #            bbox_to_anchor=(0.5, -0.5),
    #            ncol=len(log) // 2)
    ax.set_xlabel('Threshold')
    ax.set_ylabel(r'$F_1$-score')
    ax.set_xlim(0, 1)
    sns.despine()
    plt.tight_layout()
    ps(f'fscore', f)

    for n, (name, d) in enumerate(log.items()):
        title = name + ' (reduced)' if model == 'aircraft' else name
        if n > 0:  # only need one
            lab = False

        f = thres_graph(G, pos, d['thres'], title, withlabels=lab)
        ps(f'sensitivitynet_{name}', f)

    if not args.save:
        plt.show()
