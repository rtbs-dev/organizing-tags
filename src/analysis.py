import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
import statsmodels.api as sm
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import AxesGrid
from adjustText import adjust_text
from num2tex import num2tex
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import precision_recall_fscore_support
from functools import partial
import pandas as pd
import seaborn as sns
from scipy.stats import entropy
from scipy.optimize import minimize_scalar
from tqdm.autonotebook import tqdm
from cycler import cycler

from src.tools import local_and_global_consistency, softmax


def draw_G(G, pos, fn=None, fp=None, title=None, withlabels=False, **kws):
    # plt.figure(figsize=(10, 6))
    legend = kws.get('legend', True)
    leg = dict()
    if legend:
        leg['FN'] = 'FN'
        leg['FP'] = 'FP'
        leg['T'] = 'T'
    # ns = 50. if withlabels else 10.
    # nc = 'w' if withlabels else 'xkcd:slate'
    nc = 'xkcd:slate'
    nx.draw_networkx_nodes(G, pos=pos, node_color=nc,
                           with_labels=False, clip_on=False, **kws)
    nx.draw_networkx_edges(G, pos=pos, label=leg.get('T'),
                           with_labels=False, clip_on=False,
                           edge_color='dodgerblue', **kws)

    if withlabels:
        # plt.margins(0.3)
        lab_list = nx.draw_networkx_labels(
            G, pos=pos, clip_on=False,
            labels=nx.get_node_attributes(G, 'item'),
            **kws
        )
    if fp is not None:
        fp_edges = nx.draw_networkx_edges(fp, pos=pos,
                                          edge_color='#4dac26', width=1.5,
                                          label=leg.get('FP'), alpha=.5, **kws)
        if fp_edges is not None:
            #     try:
            #         [i.set_zorder(0) for i in fp_edges]
            #     except TypeError:
            fp_edges.set_zorder(0)

    if fn is not None:
        fn_edges = nx.draw_networkx_edges(fn, pos=pos,
                                          edge_color='#d01c8b', width=1.5,
                                          label=leg.get('FN'), alpha=.5, **kws)
        if fn_edges is not None:
            #     [i.set_zorder(0) for i in fn_edges]
            fn_edges.set_zorder(10)

    if (fn or fp) and legend:
        plt.legend()

    if title is not None:
        plt.title(title)

    plt.axis('off')
    if withlabels:
        adjust_text(list(lab_list.values()), lim=100)


def row_thres(X, tol=0.):
    a = X.copy()
    for i, row in enumerate(a):
        kde = sm.nonparametric.KDEUnivariate(row)
#         dist = sm.
        kde.fit(bw=1./(2.*X.shape[0]), kernel='gau', fft=False)  # nyquist
#         plt.plot(kde.density)
        plt.plot(kde.support, kde.cdf, color='b', alpha=.2)
        kneedle = KneeLocator(kde.support, kde.cdf,
                              S=1., curve='convex', direction='increasing')
        plt.axvline(kneedle.knee+tol)
        a[i, :] = np.where(row > kneedle.knee+tol, 1., 0.)
    return a


def all_thres(X, tol=0., pct_thres=95, plot=True, S=10.):
    if pct_thres is None:
        a = np.sort(X.flatten()[X.flatten() > 0.])
    #     kde = sm.nonparametric.KDEUnivariate(a)
    # #         dist = sm.
    #     kde.fit(bw=1./(2.*X.shape[0]), kernel='gau', fft=False)  # nyquist
    #     kneedle = KneeLocator(kde.support, kde.cdf,
    #                           S=S, curve='convex', direction='increasing')
    #     # thres = kneedle.knee+tol
    #     thres = X.flatten()[np.abs(X.flatten() - kneedle.knee).argmin()]

        kneedle = KneeLocator(a, np.linspace(0, 1, len(a), endpoint=False),
                              S=S, curve='convex', direction='increasing')
        # thres = kneedle.knee + tol
        thres = kneedle.xd[np.argmax(kneedle.yd)+1]

        if plot:
            ax = plt.gca()
            # ax.plot(kde.support, kde.cdf)
            plt.plot(a, np.linspace(0, 1, len(a)))
            plt.plot(kneedle.xd, kneedle.yd, 'r')
            # kneedle.plot_knee_normalized()
            ax.axvline(thres, color='g')
    else:
        assert 0 <= pct_thres <= 100, 'percentiles must be between [0,100]'
        thres = np.percentile(X, pct_thres)
    print(f'Threshold = {thres:.4e}')
    return thres, np.where(X > thres, 1., 0.)


def draw_tax(D):
    D.graph.setdefault('graph', {})['rankdir'] = 'LR'

    subg = [D.subgraph(c) for c in nx.weakly_connected_components(D)]
    fig = plt.figure(tight_layout=True, figsize=(15, 20))
    n_rows = 1+len(subg)//2+len(subg) % 2
    print(len(subg))
    gs = gridspec.GridSpec(n_rows, 2,
                           height_ratios=[3]+(n_rows-1)*[1])

    for n, d in enumerate(sorted(subg, key=len, reverse=True)):
        if n == 0:
            ax = fig.add_subplot(gs[:n+1, :])
        else:
            ax = fig.add_subplot(gs[(n+1)//2, (n+1) % 2])
        nx.draw_networkx(d, ax=ax,
                         pos=nx.drawing.nx_pydot.pydot_layout(d, prog='dot'),
                         node_size=0, edge_color='dodgerblue')
        plt.axis('off')


def compare_mats(A, labels=None, **others):
    # TODO: Redo to allow A as None!
    if A is None:
        N = len(others.keys())
    else:
        N = len(others.keys()) + 1

    # N = len(others.keys())+1
    # fig = plt.figure(figsize=(3*N, 3*N//max(N-1, 1)))
    grid = AxesGrid(plt.gcf(), 111,
                    nrows_ncols=(1, N),
                    axes_pad=0.05,
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.1
                    )
    if A is not None:
        grid[0].set_title('True')
        grid[0].set_axis_off()
        grid[0].imshow(A, vmin=0, vmax=1, aspect='equal',
                       cmap='viridis', rasterized=True)
    for n, (name, mat) in enumerate(others.items(),
                                    start=N-len(others.keys())):
        grid[n].set_title(name)
        grid[n].set_axis_off()
        im = grid[n].imshow(mat, vmin=0, vmax=1, aspect='equal',
                            cmap='viridis', rasterized=True)

    if labels is not None:
        grid[0].set_axis_on()
        grid[0].set_yticks(range(list(others.values())[0].shape[0]))
        grid[0].set_yticklabels(labels)
        grid[0].set_xticks(range(list(others.values())[0].shape[0]))
        grid[0].set_xticklabels(labels, ha='right', rotation=60)
    # grid[0].tick_params(axis='x', labelrotation=30, labelright=True)
    # when cbar_mode is 'single', for ax in grid, ax.cax = grid.cbar_axes[0]
    # plt.Axes.tick_params()

    cbar = grid[-1].cax.colorbar(im)
    cbar = grid.cbar_axes[0].colorbar(im)

    cbar.ax.set_yticks(np.arange(0, 1.1, 0.5))
    cbar.ax.set_yticklabels([0, .5, 1.])


def _plot_f_iso(ax=None):
    if ax is None:
        ax = plt.gca()
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        ax.annotate('$F_1$={0:.1f}'.format(f_score),
                    xy=(1., y[45] + 0.01),
                    color='gray')
    lines.append(l)
    labels.append('iso-f1 curves')
    plt.xlabel('Recall')
    plt.ylabel('Precision')


def get_threslist(A):
    """Smooth list of thresholds by uniform quantile samples.
    Samples 0th, 50th, 55th 60th...100th quantile.
    ignores trivial/base-case entries of zero for sparse matrices.
    """
    X = A.copy()
    if isinstance(X, pd.DataFrame):
        X = A.values
    qlist = np.linspace(50, 100, num=11)
    non_z_vals = X.flatten()[X.flatten() != 0]
    return np.array([0.]+[np.percentile(non_z_vals, q) for q in qlist])


def test(A, other):
    def flat(mat):
        if isinstance(mat, pd.DataFrame):
            return mat.values.flatten()
        else:
            return mat.flatten()

    p, r, t = precision_recall_curve(flat(A), flat(other))
    aps = average_precision_score(flat(A), flat(other))
    f = 2*p[:-1]*r[:-1]/(p[:-1]+r[:-1])
    opt_pos = np.argmax(f)

    d = dict(
        x=other,
        p=p,
        r=r,
        t=np.pad(t, (1, 0), 'constant'),
        aps=aps,
        f=np.pad(f, (1, 0), 'edge'),
        opt_pos=opt_pos,
        thres=np.where(other >= t[opt_pos], 1, 0)
    )
    return d


def test_multi(A, **others):
    results = {name: test(A, B) for name, B in others.items()}
    return results


def pvr_plot(A, other, plot_name='', ax=None, store=None, **plt_kws):
    if store is None:
        assert A is not None, ("A ground-truth `A` must be passed"
                               " if no `store` is given!")
        d = test(A, other) if store is None else store
    else:
        d = store

    opt_pos = d['opt_pos']
    if ax is None:
        ax = plt.gca()
    ax.step(d['r'], d['p'],
            where='post', marker='',
            color=plt_kws.get('color'),
            ls=plt_kws.get('linestyle'))
    aps_report = f"\nAPS = {d['aps']:.2f}"
    ax.plot(d['r'][opt_pos], d['p'][opt_pos],
            label=plot_name+aps_report,
            clip_on=False, zorder=20, **plt_kws)
    label = ax.annotate(
        r'$\sigma^*={0:.2g}$'.format(num2tex(d['t'][opt_pos])),
        (d['r'][opt_pos], d['p'][opt_pos] + 0.02),
        horizontalalignment='right', clip_on=False
    )
    d['pyplot_text'] = label
    return d


def compare_pvr(A, store=None, **others):
    if store is None:
        assert A is not None, ("A ground-truth `A` must be passed"
                               " if no `store` is given!")
        results = test_multi(A, **others)
    else:
        results = store

    pvr = partial(pvr_plot, A)

    plt.figure(figsize=(4, 5))
    _plot_f_iso()
    sns.despine()
    # TODO make this part variadic...
    custom_cycler = (cycler(color=sns.color_palette(n_colors=len(results)))
                     + cycler(ls=['-', ':', '--', '-.'])
                     + cycler(marker=['o', 's', 'X', 'd']))
    plt.gca().set_prop_cycle(custom_cycler)
    # results = {name: pvr(B, plot_name=name)
    #            for name, B in others.items()}
    for style, (name, d) in zip(custom_cycler, results.items()):
        pvr(d['x'], store=d, plot_name=name, **style)
    plt.legend(loc='upper center',
               bbox_to_anchor=(0.5, -0.2),
               ncol=len(results)//2)
    plt.tight_layout()

    plt.gca().set_aspect('equal')
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    adjust_text([i['pyplot_text'] for i in results.values()])

    # return results


def test_clf(seed_labels, y_true, A, true_prob=None, avg='micro',
             clf=local_and_global_consistency, kl_opt=False):
    """Test a semi-supervised node classification against some known
    node -> label mapping

    Classification metrics returned are micro-averaged over cateories.

    Parameters
    ----------
    seed_labels: dict
        contains initial "seed" nodes and their seed_labels
    y_true: pandas.Series
        must have a defined (ordered) pandas.CategoricalDtype
    A: array
        pandas matrix to build/classify graph (rows/columns are node names).
    clf: function
        one of the networkx.node_classification algorithms. Defaults to
        `local_and_global_consistency`.
    """
    t = get_threslist(A)
    cat = pd.CategoricalDtype(y_true.cat.categories, ordered=True)

    p, r, f, kl_vals, probs = [], [], [], [], []

    for ti in t:
        A_i = A.where(A > ti, other=0)
        G = nx.from_pandas_adjacency(A_i, create_using=nx.Graph)

        for k, v in seed_labels.items():
            if k in G.node.keys():
                G.node[k]['label'] = v

        pred_labels, pred = clf(G)

        if true_prob is not None:

            def kl_div(λ):
                prob = softmax(λ*pred[true_prob.columns])
                kl = entropy(true_prob.T, prob.T)
                return np.nansum(kl)

            opt_kl_temp = minimize_scalar(kl_div)

            K = opt_kl_temp['x']
            # K = 500.
            pred_prob = softmax(K*pred[true_prob.columns])

            kl_vals.append(entropy(true_prob.T, pred_prob.T))
            probs.append(pred_prob)

        y_pred = pd.Series(pred_labels).astype(cat)
        pi, ri, fi, _ = precision_recall_fscore_support(
            y_true.cat.codes, y_pred.cat.codes,
            average=avg,
            labels=[0, 1, 2]
        )
        p.append(pi)
        r.append(ri)
        f.append(fi)

    if kl_opt:
        opt_pos = np.argmin([np.sum(i) for i in kl_vals])
    else:
        opt_pos = np.argmax(f)

    d = dict(
        x=A,
        p=np.array(p),
        r=np.array(r),
        t=t,
        f=np.array(f),
        opt_pos=opt_pos,
        thres=A.where(A > t[opt_pos], other=0),
        kl_vals=np.array(kl_vals),
    )

    d['aps'] = (np.diff(d['r'])*d['p'][1:]).sum()

    if true_prob is not None:
        d['probs'] = probs[opt_pos]

    return d


def test_multi_clf(seed_labels, y_true, test_kws=None, **As):
    kws = dict(
        true_prob=None,
        avg='micro'
    )
    kws.update(test_kws)
    results = {
        name: test_clf(
            seed_labels, y_true, B, **kws
        ) for name, B in tqdm(As.items())
    }
    return results


def test_kl(seed_labels, true_prob, A, clf=local_and_global_consistency):
    t = get_threslist(A)
    # cat = pd.CategoricalDtype(y_true.cat.categories, ordered=True)

    kl_vals, probs, temps = [], [], []
    for ti in t:
        A_i = A.where(A > ti, other=0)
        G = nx.from_pandas_adjacency(A_i, create_using=nx.Graph)
        for k, v in seed_labels.items():
            if k in G.node.keys():
                G.node[k]['label'] = v

        pred_labels, pred = local_and_global_consistency(G)

        def kl_div(λ):
            prob = softmax(λ*pred[true_prob.columns])
            kl = entropy(true_prob.T, prob.T)
            return kl.sum()

        opt_kl_temp = minimize_scalar(kl_div)

        K = opt_kl_temp['x']
        pred_prob = softmax(K*pred[true_prob.columns])

        kl_vals.append(entropy(true_prob.T, pred_prob.T))
        temps.append(K)
        probs.append(pred_prob)

    opt_pos = np.argmin([np.sum(i) for i in kl_vals])

    d = dict(
        x=A,
        kl_vals=np.array(kl_vals),
        temps=np.array(temps),
        t=t,
        probs=probs[opt_pos],
        opt_pos=opt_pos,
        thres=A.where(A > t[opt_pos], other=0)
    )
    return d


def test_multi_kl(seed_labels, true_prob, **As):
    results = {
        name: test_kl(
            seed_labels, true_prob, B
        ) for name, B in As.items()
    }
    return results


class Record:

    def __init__(self, x_true=None):
        self.x = x_true

        self.log = None
        self.fig = None
        self.ax = None
        self.progress = None
        self.showtext = None
        self.state = None

    def err_and_mat(self, n_iter, x, text='', val=None):
        if self.log is None:
            self.log = []

        if self.fig is None:
            self.fig, self.ax = plt.subplots(ncols=2)
            self.progress, = self.ax[0].plot(1.)
            self.showtext = self.ax[0].text(
                1., 1., text,
                horizontalalignment='right',
                verticalalignment='top',
                transform=self.ax[0].transAxes
            )
            self.state = self.ax[1].imshow(x)
            self.fig.show()

        if self.x is not None:
            self.ax[0].set_title('Error from ground Truth')
            self.log += [np.linalg.norm(x - self.x)]
            self.progress.set_data(range(len(self.log)), self.log)

        elif val is not None:
            self.ax[0].set_title('Per-iteration Loss')
            self.log += [val]
            self.progress.set_data(
                range(len(self.log)),
                pd.Series(self.log).rolling(window=100).mean()
            )
        else:
            self.log += [0.]
            self.progress.set_data(range(len(self.log)), self.log)
        self.showtext.set_text(text)
        sns.despine()
        self.ax[1].axis('off')
        self.ax[0].relim()
        self.ax[0].autoscale_view()
        self.state.set_data(x)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
