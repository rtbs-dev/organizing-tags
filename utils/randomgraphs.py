import sys
import os
from pathlib import Path
# import random
from functools import partial
import itertools
import argparse
# import logging
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from cycler import cycler
# from textwrap import fill
from tqdm.autonotebook import tqdm
import utils.plot_utils as pu
from src.models import CensoredRW, train
from src.analysis import test_multi  # draw_G , compare_pvr, compare_mats
from src.invite import sample_censored

from src.tools import cosine_model, markov_model
from src.tools import minmax, symmetrize  # , fill_diagonal, labeled_adj

sys.path.insert(0, os.path.abspath('..'))


toplvl_directory = Path(__file__).parents[1]
net_path = toplvl_directory/'data'/'randomgraphs_serial'
latex_path = toplvl_directory/'latex'/'jmd19'/'img'/'randomgraphs'
latex_path.mkdir(parents=True, exist_ok=True)
log_path = latex_path/'logfile-randomgraphs.txt'


def gen_random_graphs(ngraphs, *sizes, rewire=0.166, meandeg=4, read_in=True):

    size_iter = itertools.cycle(sizes)

    def gname(i):
        return f'G{str(i)}_N{"-".join(map(str,sizes))}.gpickle'
    read_g = 0
    if read_in:
        for gfile in net_path.glob(gname('*')):
            # print(f'K={read_g}')
            read_g += 1
            yield nx.read_gpickle(gfile)

    for n in range(read_g, ngraphs):
        # print(f'N={n}')
        size = next(size_iter)
        G = nx.watts_strogatz_graph(size, meandeg, rewire)
        nx.write_gpickle(G, net_path/gname(n))
        yield G


def stream_data_store(args):
    for n, G in tqdm(enumerate(
        gen_random_graphs(args.graphs, *args.nodes,
                          read_in=args.readin)
    )):
        A = nx.to_numpy_array(G)
        N = G.number_of_nodes()
        # Mcore = sample_censored(G, n_walks=0,
        #                     n_obsv=args.length, steps=100, demo=False)
        for nwalks in args.nwalks:
            M = sample_censored(G, n_walks=nwalks,
                                n_obsv=args.length,
                                steps=100, demo=False, cuda=False)
            model = CensoredRW(N, sym=True)

            learning_rate = 0.1

            train(model, np.array(M), batch_size=None, compare_true=A,
                  epochs=50, callback=False, lr=learning_rate)

            res = model.P.detach().cpu().numpy()

            def prep(A): return minmax(symmetrize(A))

            mkv1 = markov_model(M, k=1)
            mkv2 = markov_model(M, k=2)
            cos = cosine_model(M)

            models = dict(
                INVITE=prep(res),
                Cosine=prep(cos),
                MC1=prep(mkv1),
                MC2=prep(mkv2),
            )

            logs = test_multi(A, **models)

            for name, log in logs.items():
                for pos, t in enumerate(log['t']):
                    d = dict(
                        graph=f'G{n}',
                        model=name,
                        aps=log['aps'],
                        f_opt=log['f'][log['opt_pos']],
                        t_opt=log['opt_pos'],
                        t=t, r=log['r'][pos], p=log['p'][pos],
                        f=log['f'][pos],
                        nwalks=nwalks,
                        nodes=N,
                    )
                    yield d


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate plot/results for Watts-Strogats random graphs. '
    )
    parser.add_argument(
        '-g', '--graphs',
        default=30, type=int,
        help='how many random graphs to test/serialize',
    )
    parser.add_argument(
        '-o', '--nodes', default=[10, 25, 50],
        type=int, nargs='+',
        help='how many nodes to generate from (can be multiple)',
    )
    parser.add_argument('-n', '--nwalks',
                        default=[10, 25, 50], type=int, nargs='+',
                        help='# of (extra) synthetic MWOs to INVITE sample'
                             "(By default, as many as needed to cover nodes)")
    parser.add_argument('-l', '--length', default=4, type=int,
                        help='length of synthetic mwos via INVITE sample')
    parser.add_argument('-r', '--seed', default=8, type=int,
                        help='(r)andom seed for numpy, for reproducibility')
    parser.add_argument('-f', '--readin', action='store_true', default=False,
                        help='whether to read existing, matching archive data')
    parser.add_argument('-s', '--save', action='store_true', default=False)

    args = parser.parse_args()

    pu.figure_setup()
    ps = partial(pu.plot_or_save, latex_path, args,
                 transparent=True)
    np.random.seed(8)

    # logging.basicConfig(
    #     filename=log_path,
    #     level=logging.INFO,
    #     format='%(asctime)s %(message)s',
    #     datefmt='%d/%m/%Y %H:%M:%S',
    # )
    # logging.info('############ NEW RUN #############')
    # logging.info("Run Parameters:\n" + str(args))

    # ######### GET DATA ########### #
    datastore = net_path/pu.args_to_str(args, skip=['readin', 'save'])
    print(datastore)
    if args.readin and datastore.is_file():
        df = pd.read_feather(datastore)
    else:
        data = stream_data_store(args)
        df = pd.DataFrame(data)
        df.to_feather(datastore)

    # keep thresholds at even values
    df = df.assign(**{r'$\sigma$': df['t'].round(decimals=1)}).rename(
        columns={
            'nwalks': r'$C$',
            'nodes': r'$N$',
            'f': r'$F_1$-score',
        }
    )

    g = sns.relplot(data=df, x=r'$\sigma$', y=r'$F_1$-score',
                    col=r'$C$',
                    row=r'$N$',
                    hue='model', kind='line', style='model',
                    facet_kws={"hue_order": ['INVITE', 'Cosine', 'MC1', 'MC2'],
                               "hue_kws": dict(
                                    linestyle=['-', ':', '--', '-.']
                               ),
                               # 'legend_out': False
                               },
                    estimator=np.median,
                    legend=False
                    # height=pu.TEXT_HEIGHT/3.,
                    # aspect=pu.TEXT_WIDTH/pu.TEXT_HEIGHT
                    )
    # g.set(ylabel='$F_1$-score')
    # g.axes[0][0].legend(fontsize=pu.label_size(),
    #                     title_fontsize=pu.font_size())
    plt.gcf().set_size_inches(*pu.figsize(columns=2,
                                          fig_height=pu.TEXT_HEIGHT / 2.))
    plt.tight_layout()
    ps('fscores')

    g = sns.relplot(data=df, x=r'$\sigma$', y='r',
                    hue='model', kind='line', style='model',
                    facet_kws={'legend_out': False},
                    estimator=np.median,
                    legend=False,
                    )
    # plt.legend(fontsize=pu.label_size(), title_fontsize=pu.font_size())
    plt.gcf().set_size_inches(*pu.figsize(columns=1,
                                          aspect=1.6))
    g.set(ylabel='Recall')
    # plt.legend(False)
    plt.tight_layout()
    ps('t_v_r')

    g = sns.relplot(data=df, x=r'$\sigma$', y='p',
                    hue='model', kind='line', style='model',
                    facet_kws={'legend_out': False},
                    estimator=np.median,
                    )
    plt.legend(fontsize=pu.label_size(), title_fontsize=pu.font_size())
    plt.gcf().set_size_inches(*pu.figsize(columns=1,
                                          aspect=1.6))
    g.set(ylabel='Precision')
    plt.tight_layout()
    ps('p_v_r')

    if not args.save:
        plt.show()
