# import sys
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial

from sklearn import preprocessing as pre
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_curve, average_precision_score

from src.invite import sample_censored
from DEPRECATED.loss import loss
from DEPRECATED.opt import job_minibatch
from src.analysis import draw_G, all_thres, Record, compare_3

from sacred import Experiment
from sacred.observers import MongoObserver

import warnings
warnings.warn("the recovery_experiment module is deprecated!",
              DeprecationWarning, stacklevel=2)

ex = Experiment('idetc19')
ex.observers.append(MongoObserver.create())


def set_style():
    # This sets reasonable defaults for font size for a figure that will go in a paper
    sns.set_context("paper")
    # Set the font to be serif, rather than sans
    sns.set(font='serif')
    # Make the background white, and specify the specific font family
    sns.set_style("white", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })


@ex.config
def model_config():
    model = 'drivetrain'
    nitems = 3  # how many tags per sample
    nwos = 500  # how many samples
    init_type = 'rand'
    adam_kws = dict(
        epochs=10,
        learning_rate=0.9,
        batch_size=5,
        reg=.01,
        decay=True,
        avg=True,
    )

    if model == 'aircraft':
        graph_labels = False
    else:
        graph_labels = True


@ex.capture
def load_data(model):
    net_path = Path('data') / 'network_models_shared'
    mtypes = ['bicycle', 'drivetrain', 'aircraft']
    assert model in mtypes, f'model must be one of {mtypes}'

    fname = net_path / f'{model}_component_model.xlsx'
    ex.add_resource(fname)

    struct = pd.read_excel(fname, index_col=0)
    return struct


@ex.capture(prefix='adam_kws')
def optimize(f, X_init, obsv,
             epochs, learning_rate, batch_size, reg, decay, avg,
             callback):
    X_opt = job_minibatch(
        f, X_init, obsv,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        reg=reg,
        decay=decay,
        avg=avg,
        callback=callback
    )
    plt.close('all')
    return X_opt


@ex.automain
def run_experiment(_run, nitems, nwos, init_type, graph_labels):
    tmp = Path('.')/'tmp'
    set_style()

    struct = load_data()
    np.save(tmp / 'true_adj.npy', struct.values)
    ex.add_artifact(tmp / 'true_adj.npy')

    G = nx.from_pandas_adjacency(struct)
    G = nx.convert_node_labels_to_integers(G, label_attribute='item')
    pos = nx.layout.kamada_kawai_layout(G)
    # pos = nx.fruchterman_reingold_layout(G)
    # pos = nx.circular_layout(G)
    print(f'C_β = {nx.average_clustering(G):.2f}')
    draw_G(G, pos, title='True Structure', withlabels=graph_labels)
    plt.savefig(tmp/'true_G.png')
    ex.add_artifact(tmp/'true_G.png')

    # print('synthesizing MWOs...')
    # m = [graph_random_walk(G, steps=100).unique()[:int(nitems)]
    #      for i in trange(int(nwos))]
    #
    #
    # print('example MWOs: ')
    # for n, i in enumerate(m[:4]):
    #     print(n, ' → '.join('{:^2}'.format(G.nodes[j]['item']) for j in i.tolist()))
    # print('\n'.join(i for i in 3*['.']))
    # print(len(m)-1, ' → '.join('{:^2}'.format(G.nodes[j]['item']) for j in m[-1].tolist()))

    m = sample_censored(G, int(nwos), int(nitems), steps=100)

    A = nx.to_numpy_array(G)
    encode = pre.MultiLabelBinarizer(classes=range(A.shape[0]))
    tags = encode.fit_transform(m)

    cos = cosine_similarity(tags.T.dot(tags))
    np.save(tmp/'cosine_adj.npy', cos)
    ex.add_artifact(tmp/'cosine_adj.npy')

    if init_type is 'cos':
        B = np.copy(cos)
    else:
        B = np.random.rand(*A.shape)
        B = B.T + B - 2*np.diag(B.diagonal())
    np.save(tmp / 'init_adj.npy', B)
    ex.add_artifact(tmp / 'init_adj.npy')

    rec = Record(x_true=A)

    def callback_wrap(rec, *args, **kwargs):
        args = list(args)
        if 'val' in kwargs.keys():
            if kwargs['val'] is not None:
                _run.log_scalar("training.loss",
                                kwargs['val'], args[0])
        if rec.log:
            _run.log_scalar('training.error', rec.log[-1], args[0])
        return rec.err_and_mat(*args, **kwargs)
    callback = partial(callback_wrap, rec)

    A_approx = optimize(loss, B, m, callback=callback)
    np.save(tmp/'invite_adj.npy', A_approx)
    ex.add_artifact(tmp/'invite_adj.npy')

    labels = struct.columns if graph_labels else None
    fig = compare_3(A, A_approx, cos, labels=labels)
    plt.savefig(tmp/'estimate_mat_comparison.png')
    ex.add_artifact(tmp/'estimate_mat_comparison.png')

    def graph_compare(A, G, X_thres, name, label):
        in_thres = nx.from_numpy_array(X_thres - A > 0)  # draw FP/FN Graph
        not_in_thres = nx.from_numpy_array(A - X_thres > 0)
        draw_G(G, pos, fp=in_thres, fn=not_in_thres,
               title=f'{name}: Recovered Structure - {label}',
               withlabels=graph_labels)
        plt.savefig(tmp / f'{name}_{label}_graph.png')
        ex.add_artifact(tmp / f'{name}_{label}_graph.png')

    def capture_results(A, X, name):
        store = dict(
            # X=X,
        )

        ### Un-Supervised Metric (knee-finding heuristic) ###
        plt.figure()
        knee, A_thres = all_thres(X, pct_thres=None, plot=True)  # find knee
        plt.savefig(tmp/f'{name}_knee.png')
        ex.add_artifact(tmp/f'{name}_knee.png')

        graph_compare(A, G, A_thres, name, 'knee')

        f_knee = metrics.fbeta_score(A.flatten(), A_thres.flatten(), 1.)
        print(f'{name} - Knee F_1 = {f_knee:.3f}')  # get knee-based fscore

        unsup = dict(
            knee = knee,
            # X_knee = A_thres,
            fscore_knee = f_knee
        )
        store['unsupervised'] = unsup

        ### Supervised Optimum (Best F1-Score) ###
        p_, r_, t_ = precision_recall_curve(A.flatten(), X.flatten())
        aps_ = average_precision_score(A.flatten(), X.flatten())
        f_ = 2 * p_[:-1] * r_[:-1] / (p_[:-1] + r_[:-1])
        ts_ = t_[np.nanargmax(f_)]  # best threshold for f1-score
        print(f'{name} - Opt. F_1 = {np.nanmax(f_):.3f}')  # get knee-based fscore

        B_thres = np.where(X>=ts_, 1., 0.)
        graph_compare(A, G, B_thres, name, 'f-score')

        sup = dict(
            # X_opt=B_thres,
            precision=p_,
            recall=r_,
            thres=t_,
            fscores=np.where(np.isnan(f_), 0, f_),
            thres_opt=ts_,
            fscore_opt=np.nanmax(f_),
            aps=aps_,
        )
        store['supervised'] = sup

        T = nx.maximum_spanning_tree(G)
        pathfind = nx.to_numpy_array(nx.maximum_spanning_tree(nx.Graph(X)))
        C_thres = np.where(pathfind > 0, 1, 0)
        f_pf = metrics.fbeta_score(nx.to_numpy_array(T).flatten(),
                                   C_thres.flatten(), 1.)
        graph_compare(A, T, C_thres, name, 'pathfinder')
        pf = dict(
            # X_pf = C_thres,
            fscore_pf = f_pf,
        )
        store['pathfinder'] = pf
        # pprint.pprint(store)
        return store

    print('\nCalculating metrics for model...')
    _run.info['invite'] = capture_results(A, A_approx, 'invite')

    print('\nCalculating metrics for naive...')
    _run.info['cosine'] = capture_results(A, cos, 'cosine')

    ivt = _run.info['invite']['supervised']
    aps_ivt = ivt['aps']
    # _run.log_scalar('invite_avg_precision_score', aps_ivt)

    cos = _run.info['cosine']['supervised']
    aps_cos = cos['aps']
    # _run.log_scalar('cosine_avg_precision_score', aps_cos)

    plt.figure()

    plt.step(ivt['recall'], ivt['precision'], alpha=0.2, where='post')
    plt.fill_between(ivt['recall'], ivt['precision'], alpha=0.2, step='post')

    plt.step(cos['recall'], cos['precision'], alpha=0.2, where='post')
    plt.fill_between(cos['recall'], cos['precision'], alpha=0.2, step='post')
    plt.title(f'Avg. Precision (INVITE) = {aps_ivt:.3f}\n' +
              f'Avg. Precision (Cosine) = {aps_cos:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    sns.despine()

    plt.savefig(tmp/'p_vs_r.png')
    ex.add_artifact(tmp/'p_vs_r.png')

    return aps_ivt, aps_cos
    # plt.figure()
    # plt.imshow((cos_thres - A))
    # plt.show(block=False)
    #
    # in_cos     = nx.from_numpy_array(cos_thres - A > 0)
    # not_in_cos = nx.from_numpy_array(A - cos_thres > 0)
    #
    # draw_G(G, pos, fp=in_cos, fn=not_in_cos)
    # plt.show()
