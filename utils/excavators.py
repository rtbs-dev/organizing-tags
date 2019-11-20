__author__ = 'Thurston Sexton'


from pathlib import Path
import sys
import os
import argparse
from functools import partial
import re
import logging
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ternary
import seaborn as sns
from cycler import cycler
import nestor.keyword as kex
from nestor.datasets import load_excavators
from src.models import CensoredRW, train
from src import tools
from src.analysis import draw_G, test_multi_clf, compare_mats
import utils.plot_utils as pu
sys.path.insert(0, os.path.abspath('..'))

toplvl_directory = Path(__file__).parents[1]
net_path = toplvl_directory/'data'/'mine_public'
latex_path = toplvl_directory/'latex'/'jmd19'/'img'/'excavators'
latex_path.mkdir(parents=True, exist_ok=True)
log_path = latex_path/'logfile-excavators.txt'


def extract_walk(raw_text, vocab):
    """extract keyworded tags from `raw_text`, preserving word-order.
    """
    thes_dict = vocab.alias.to_dict()
    substr = sorted(thes_dict, key=len, reverse=True)
    flag = r'\b(' + '|'.join(map(re.escape, substr)) + r')\b'
    mult = raw_text.str.findall(flag)
    return mult


def filter_tag_occurrences(input_df, freq=10, ntags=3, topn=None):
    """ TODO: integrate these helper functions with the `nestor` lib
    We want to restrict analysed tags to the top `topn` (if given), s.t.
        - each tag occurs at least `freq` times in the corpus, and
        - each document has at least `ntags` tags.

    Because these are not independent constraints, this routine will iterate
    their enforcement until convergence.

    Parameters
    ----------
    input_df: pd.Dataframe
        binary tag dataframe, output from `nestor` library.
    freq: int
        minimum number of occurences for a tag to be considered
    l: int
        minimum number of tags for a document to be considered.
    topn: int, optional
        restrict allowed tags to the `topn` most frequent

    Returns
    -------
    pd.Dataframe, filtered with the set constraints. NOTE there may be fewer
    than `topn` tags returned, either due to high `freq`, or to no set of
    exactly `topn` tags meeting these constraints.
    """
    def filter_tag_freq(input_df, freq=freq):
        """remove tags with fewer than `freq` ocurrences"""
        return input_df.loc[:, input_df.sum() > freq]

    def filter_tag_len(input_df, ntags=ntags):
        """remove documents with fewer than `ntags` tags"""
        return input_df.loc[input_df.sum(axis=1) >= ntags, :]

    def filter_tag_shave(input_df):
        """remove tags with the current lowest frequency"""
        msk = input_df.sum() > input_df.sum().min()
        return input_df.loc[:, msk]

    temp_df = (
        input_df
        .pipe(filter_tag_freq, freq=ntags)
        .pipe(filter_tag_len, ntags=ntags)
    )
    topfilt = False if topn is None else temp_df.columns.shape[0] > topn
    while (
        (temp_df.sum(axis=1) < ntags).any()
    ) or (
        (temp_df.sum() < freq).any()
    ) or topfilt:  # repeat until convergence
        temp_df = (
            temp_df
            .pipe(filter_tag_shave)
            .pipe(filter_tag_freq)
            .pipe(filter_tag_len)
        )
        topfilt = False if topn is None else temp_df.columns.shape[0] > topn
    return temp_df


def filter_tag_names(input_df, namelist):
    filt = (input_df
            .loc[:, ~input_df.columns.duplicated()]
            .drop(namelist, axis=1))
    return filt


def row_norm(input_df):
    """make occurence dataframe row-stochastic"""
    return input_df.div(input_df.sum(axis=1), axis=0)


def mask_confusion(input_df, ratio_limit=0.6):
    """ remove labels where one does not dominate (>half of prob. mass)"""
    return input_df.mask(~(input_df > ratio_limit).any(axis=1))


def labeled_adj(names, X):
    return pd.DataFrame(X, columns=names, index=names)


def plot_tern(probs, subsys, title, color_dict,
              ax=None, legend=False, info=True, markersize=10):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 4))
    else:
        fig = plt.gcf()
    ax.set_aspect('equal')
    tax = ternary.TernaryAxesSubplot(ax=ax, scale=1.)

    tax.set_title(f"{title}", fontsize=pu.font_size())
    tax.boundary()
    tax.gridlines(multiple=.2, color="black")
    if info:
        tax.ticks(axis='lbr', multiple=.2, tick_formats="%.1f", offset=0.05,
                  fontsize=pu.font_size())

        tax.left_axis_label(f"{probs.columns[2]}", offset=.25,
                            fontsize=pu.font_size())
        tax.right_axis_label(f"{probs.columns[1]}", offset=.25,
                             fontsize=pu.font_size())
        tax.bottom_axis_label(f"{probs.columns[0]}", offset=.28,
                              fontsize=pu.font_size())
    # scats = []
    for i in subsys.unique():
        if pd.isna(i):
            tax.scatter(
                probs[subsys.isna()].values,
                color=color_dict[i],
                label='Not Classified',
                s=markersize,
            )
            # scats.append(scat)
        else:
            tax.scatter(
                probs[subsys.values == i].values,
                color=color_dict[i],
                zorder=10,
                label=i,
                s=markersize,
            )
            # scats.append(scat)
    if legend:
        tax.legend(loc=legend, fontsize=pu.label_size())

    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')
    return fig


def mstree_plot(A_thres, title=None, ax=None):
    """plot maximum spanning tree with layered edges for viz"""
    if ax is None:
        ax = plt.gca()
    G = nx.from_pandas_adjacency(
        A_thres,
        create_using=nx.Graph
    )
    G = nx.convert_node_labels_to_integers(G, label_attribute='item')
    D = nx.maximum_spanning_tree(G)

    nontree_edges = nx.from_numpy_array(
        nx.to_numpy_array(G) - nx.to_numpy_array(D) > 0,
        create_using=nx.Graph
    )
    pos = nx.layout.kamada_kawai_layout(D)
    if title is not None:
        ax.set_title(title)
    draw_G(D, pos, fp=nontree_edges,
           withlabels=True, font_size=8., font_family='serif',
           legend=False, ax=ax, node_size=20.)

    #     print(f'C_β = {nx.average_clustering(G):.2f}')
    ax.axis('off')
    ax.set_clip_on(False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Generate plot/results for Excavator case study'
    )
    parser.add_argument('-n', '--ntags', default=3, type=int,
                        help='minimum number of tags to require in a document')
    parser.add_argument('-f', '--freq', default=5, type=int,
                        help='minumum tag occurrence frequency')
    parser.add_argument('-p', '--topn', default=50, type=int,
                        help='max. # of tags to allow')
    parser.add_argument('-t', '--thres', default=60, type=int,
                        help='decision bound, for tag prob mass (0<thres<100)')
    parser.add_argument('-s', '--save', action='store_true', default=False)
    args = parser.parse_args()
    ps = partial(pu.plot_or_save, latex_path, args,
                 transparent=True)

    pu.figure_setup()
    np.random.seed(5)

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%d/%m/%Y %H:%M:%S',
    )
    logging.info('############ NEW RUN #############')
    logging.info("Run Parameters:\n"+str(args))
    # ########## GET DATA ########### #
    df = load_excavators(cleaned=True)  # load the excavators dataset

    # retrieve annotated token -> tag mapping
    vocab = (
        pd.read_excel(
            net_path / 'vocab_aus.xlsx',
            index_col=0,
            dtype={'alias': str},
        )
        .pipe(lambda input_df: input_df.set_index(input_df.index.map(str)))
        .fillna('')
        .reset_index()
        .drop_duplicates(subset=['tokens'])
        .set_index('tokens')
    )

    # merge and cleanse NLP-containing columns of the data
    known_repl = {
        'l h': 'lh',
        'r h': 'rh',
        'a c': 'ac',
        'air con': 'ac',
        'rf': 'right_front'
    }
    nlp_select = kex.NLPSelect(
        columns=['OriginalShorttext'],
        special_replace=known_repl,
    )
    raw_text = nlp_select.transform(df)
    # raw_text, with token-->alias replacement
    replaced_text = kex.token_to_alias(raw_text, vocab)

    which_types = [  # only items/object within system, not actions/properties
        'I',
        # 'P', 'S',
        # 'U',
    ]

    tex = kex.TokenExtractor()
    toks = tex.fit_transform(replaced_text)
    tag_df = kex.tag_extractor(tex, replaced_text, vocab_df=vocab)[which_types]

    filt_tags = (
        tag_df
        .droplevel(0, axis=1)
        .pipe(filter_tag_names, ['position', 'right'])
        .pipe(filter_tag_occurrences,
              ntags=args.ntags,
              freq=args.freq,
              topn=args.topn)
    )
    voc_nodes = filt_tags.columns

    walks = extract_walk(
        replaced_text[filt_tags.index], vocab[vocab.index.isin(
            filt_tags.columns)]
    )
    logging.info(
        'Tag RW length breakdown:\n'
        + walks
        .apply(len)
        .describe()
        .round(2)
        .to_csv(encoding='utf8', sep='\t')
    )
    int_map = {i: k for k, i in enumerate(voc_nodes)}
    # at last, the actual "random walks" along the "tag graph"
    m = [np.unique([int_map[j] for j in i]) for n, i in walks.iteritems()]

    # #############  CLASSIFICATION STUDY ############### #
    # top 3 "major subsystem" labels in the cleaned dataset
    # from Hodkiewicz et al
    subsys_cats = pd.CategoricalDtype(
        ordered=True,
        categories=df.MajorSystem.value_counts()[:3].index.tolist()
    )
    subsys_breakdown = {  # major subsystem occurrences
        sys: filt_tags[df.MajorSystem.str.contains(sys, na=False)].sum(axis=0)
        for sys in df.MajorSystem.value_counts()[:3].index.tolist()
    }

    tag_probs = (  # tag probability given subsystem label
        pd.DataFrame()
        .assign(**subsys_breakdown)
        .pipe(row_norm)  # make class counts into "belonging" fraction
    )

    subsys = (
        tag_probs
        .pipe(mask_confusion, ratio_limit=args.thres/100.)
        .idxmax(axis=1)
        .astype(subsys_cats)
    )

    color_dict = dict(zip(  # good colors for the subsystems
        subsys.cat.categories.tolist() + [np.nan],
        ['dodgerblue', 'xkcd:rust', 'g', 'k']
    ))

    axes = tag_probs.plot.bar(subplots=True,
                              figsize=pu.figsize(columns=2, aspect=2.),
                              legend=False)
    for lab in axes[2].get_xticklabels():
        sys = subsys[lab.get_text()]
        lab.set_color(color_dict[sys])
    plt.tight_layout()
    sns.despine()
    ps('tag_multinomial')

    # ################## Calculate INVITE Distances ################ #
    logging.info('training...')
    N = voc_nodes.shape[0]
    # uses pytorch + ADAM
    model = CensoredRW(N, cuda=False, sym=False)
    learning_rate = 0.1

    train(model, np.array(m), batch_size=min(500, len(m)),
          epochs=50, callback=True, lr=learning_rate)

    logging.info('Done!')

    # ########## COMPARISONS ############# #

    def prep(A): return labeled_adj(
        voc_nodes, tools.minmax(tools.symmetrize(A)))

    res = tools.fill_diagonal(model.P.detach().cpu().numpy(), 0)

    mkv1 = tools.markov_model(m, k=1)  # order 1, via `pomegranate`
    mkv2 = tools.markov_model(m, k=2)  # order 2, via `pomegranate`
    # mkv2 = tools.hidden_markov(m, max(map(len, m)), N)
    cos = tools.cosine_model(m)  # cosine similarity, via `sklearn`

    models = dict(
        INVITE=prep(res),
        Cosine=prep(cos),
        MC1=prep(mkv1),
        MC2=prep(mkv2),
    )

    fig = plt.figure(figsize=pu.figsize(columns=2, aspect=4.1))
    compare_mats(
        None,
        # labels=models['INVITE'].columns,
        **models
    )
    ps('matrix_comparison', fig=fig)

    # ############## Semi-supervised Classification ############## #
    toplvl = dict([
        ('hydraulic', 'Hydraulic System'),
        # ('hose',       'Hydraulic System'),
        # ('pump',       'Hydraulic System'),
        # ('compressor', 'Hydraulic System'),
        ('bucket', 'Bucket'),
        # ('tooth',      'Bucket'),
        # ('lip',        'Bucket'),
        # ('pin',        'Bucket'),
        ('engine', 'Engine'),
        # ('filter',     'Engine'),
        # ('fan',        'Engine'),
    ])

    log = test_multi_clf(toplvl, subsys,
                         test_kws=dict(true_prob=tag_probs,
                                       kl_opt=False, avg='weighted'),
                         **models)
    report = pd.DataFrame({
        name: [
            d['f'][d['opt_pos']],
            np.nansum(d['kl_vals'][d['opt_pos']]),
            np.nanmean(d['kl_vals'][d['opt_pos']]),
            np.nanstd(d['kl_vals'][d['opt_pos']]),
        ]
        for name, d in log.items()
    }, index=[
        r'$F_1^*$',
        r'$\Sigma KL$',
        r'$\mu_{KL}$',
        r'$\sigma_{KL}$'
    ]).T

    logging.info('LaTeX Report:\n ' + report.to_latex(escape=False))

    custom_cycler = (cycler(color=sns.color_palette(n_colors=4))
                     + cycler(ls=['-', ':', '--', '-.']))

    fig, ax = plt.subplots(ncols=2, sharey=False,
                           figsize=pu.figsize(columns=2, aspect=3.))
    for axis in ax:
        axis.set_prop_cycle(custom_cycler)
    for name, d in log.items():
        ax[0].plot(d['t'], d['f'], label=name)
        ax[0].set_xlabel('Threshold')
        ax[0].set_ylabel(r'$F_1$-score')
        ts = (
            pd.DataFrame(d['kl_vals'], index=d['t'])
            .reset_index()
            .melt(id_vars='index')
            .astype(float)
        )
        # sns.lineplot(data=ts, x='index', y='value',
        #              label=name,
        #              estimator=np.nanmedian,
        #              # err_style='bars',
        #              ci=None,
        #              ax=ax[1])
        # ax[1].set_yscale('log')

        mean = np.nansum(d['kl_vals'], axis=1)
        ax[1].plot(d['t'], mean, label=name)
        ax[1].legend(loc=0, fontsize=pu.label_size())
        ax[1].set_xlabel('Threshold')
        ax[1].set_ylabel('Total KL-Div.')

        logging.info(
            name + '\n \t'
            + f'Total KLD:\t {np.nansum(d["kl_vals"][d["opt_pos"]]):.4f} \n \t'
            + f'Mean KLD:\t {np.nanmean(d["kl_vals"][d["opt_pos"]]):.4f} \n \t'
            + f'Stdv KLD:\t {np.nanstd(d["kl_vals"][d["opt_pos"]]):.4f} \n \t'
            + f'F1:\t {np.nanstd(d["f"][d["opt_pos"]]):.4f} \n \t'
        )

    sns.despine()
    plt.tight_layout()

    ps('F1_KL', fig=fig)

    fig, ax = plt.subplots(ncols=2, sharey=True,
                           figsize=pu.figsize(columns=2, aspect=3.))
    for axis in ax:
        axis.set_prop_cycle(custom_cycler)

    for name, d in log.items():
        ax[0].plot(d['t'], d['p'], label=name)
        ax[0].legend(loc=0, fontsize=pu.label_size())
        ax[0].set_xlabel('Threshold')
        ax[0].set_ylabel('Precision')
        ax[1].plot(d['t'], d['r'], label=name)
        #     ax[1].legend(loc=0)
        ax[1].set_xlabel('Threshold')
        ax[1].set_ylabel('Recall')
    #     plt.xlim(0,1)
    sns.despine()
    plt.tight_layout()

    ps('precision_recall', fig=fig)

    # ############ Ternary Plots ########### #
    f, ax = plt.subplots(ncols=3, nrows=2,
                         figsize=pu.figsize(columns=2, aspect=1.6),
                         gridspec_kw=dict(hspace=.3, wspace=.2))
    plot_tern(tag_probs, subsys, 'True\n', color_dict,
              ax=ax.flatten()[0], info=True)

    for n, (name, d) in enumerate(log.items()):
        axis = ax.flatten()[n + 1]
        legend = False
        if n == 3:
            legend = (1.3, .3)
        plot_tern(d['probs'], subsys, name,
                  color_dict, ax=axis, legend=legend, info=False)
        axis.text(
            .7, .7, f'Σ(KLD)\n  ={np.nansum(d["kl_vals"][d["opt_pos"]]):.1f}',
            fontsize=pu.font_size(),
        )
    for axis in ax.flatten():
        axis.set_aspect(1.)

    ax.flatten()[-1].axis('off')
    ps('ternary', fig=f)

    # ############### Network Plots ################# #
    for model in log:
        f = plt.figure(figsize=pu.figsize(
            fig_height=pu.TEXT_HEIGHT / 2., columns=1))
        mstree_plot(log[model]['thres'], title=model)
        ps(f'{model}_network', fig=f)

    if not args.save:
        plt.show()
