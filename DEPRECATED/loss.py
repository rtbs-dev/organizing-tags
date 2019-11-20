import autograd.numpy as np
import warnings
from functools import partial
__all__ = [
    'P_i',
    'P_a',
    'P_m',
    'loss',
    'loss_i',
]

warnings.warn("the `loss` module is deprecated!", DeprecationWarning,
              stacklevel=2)


def P_i(T, a, idx):
    """ Probability of absorbtions given an observed chain:

    We need to partition the transition matrix
        $$ T =
        \begin{pmatrix}
         Q & R \\
         0 & I
        \end{pmatrix}
        $$
    where:
        $Q$: the non-absorbing transitions,
        $R$: non-absorbing to absorbing transitions
    Then, probability of being absorbed is given as
        $$P = (I-Q)^{-1} R$$

    In this case, we only want the probability of transitioning from the
    most recent state to the current absorbing state.
    """

    a_trans = np.array(a)[0:idx]  # visited
    a_absrb = np.array(a)[idx:len(a) - idx]  # not visited

    Q = T[a_trans, :][:, a_trans]
    R = T[a_trans, :][:, a_absrb]
    I = np.identity(Q.shape[0])

    P = np.dot(np.linalg.pinv(I-Q), R)

    return P[-1, 0]  # ...from previous state (P[-1,:] by construction) into next


def P_a(T, a):
    """ Calculate the log-likelihood of a transition matrix $T$, given censored
    observed INVITE sequence $a$.
    """

    frontload_a = list(a) + list(set(range(T.shape[0])) - set(a))
    like_i = partial(P_i, T, frontload_a)

    return -1*np.sum([np.log(like_i(idx)) for idx in range(1, len(a)-1)])


def _symmetrize(a):  # assumes 0-diags

    bott = np.tril(a) + np.tril(a).T
    top = np.triu(a) + np.triu(a).T
    # return (bott+top)/2. + infs
    return np.fmax(bott, top)


def _softmax(a, axis=None):
    a = a - a.max(axis=axis, keepdims=True)
    infs = np.diag(a.shape[0] * [-np.inf])

    y = np.exp(a + infs)
    return y / y.sum(axis=axis, keepdims=True)


def loss_i(m, idx, A, reg=1e-2):
    """Per-iteration objective function for use in ASGD"""

    T = _softmax(_symmetrize(A), axis=1)

    # for it in range(1, 4):  # 2x stochastic; Sinkhorn, 1964
    #     T = _softmax(T + np.diag(A.shape[0]*[-np.inf]), axis=it % 2)

    like = P_a(T, m[idx])
    # penalty = (1. / len(m)) * np.linalg.norm(A)  # Frob-norm
    penalty = (1. / len(m)) * np.abs(A).sum(axis=0).max()  # L1-norm

    return like + reg*penalty


def loss(m, A, reg=1e-2, sym=True):
    """Per-batch objective function for use in ASGD"""

    # infs = np.diag(A.shape[0] * [-np.inf])
    if sym:
        T = _softmax(_symmetrize(A), axis=1)
    else:
        T = _softmax(A, axis=1)
    # T = sinkhorn(A)
    f_a = partial(P_a, T)
    # print(m)
    like = np.sum(np.array([f_a(a) for a in m]))
    # penalty = (1. / np.max([1., len(m)])) * np.linalg.norm(A)  # Frob-norm
    penalty = (1. / np.max([1., len(m)])) * np.abs(A).sum(axis=0).max()  # L1-norm

    return like + reg*penalty


def P_m(m, A):
    """ Calculate total log-likelihood of a a transition matrix $T$, given
    a list of M censored INVITE sequence observations, ${a_1, a_2,\cdots a_M}$
    """
    like_a = partial(P_a, A)
    return np.sum([like_a(a) for a in m])


def sinkhorn(P):
    """Fit the diagonal matrices in Sinkhorn Knopp's algorithm
    """


    N = P.shape[0]
    max_thresh = 1 + 1e-3
    min_thresh = 1 - 1e-3
    _iterations = 0
    _max_iter = 100
    # _stopping_condition = None
    # Initialize r and c, the diagonals of D1 and D2
    # and warn if the matrix does not have support.
    r = np.ones((N, 1))
    pdotr = np.dot(P.T, r)
    # total_support_warning_str = (
    #     "Matrix P must have total support. "
    #     "See documentation"
    # )
    # if not np.all(pdotr != 0):
    #     warnings.warn(total_support_warning_str, UserWarning)
    #
    c = 1 / pdotr
    pdotc = np.dot(P, c)
    # if not np.all(pdotc != 0):
    #     warnings.warn(total_support_warning_str, UserWarning)
    #
    r = 1 / pdotc
    # del pdotr, pdotc

    P_eps = P
    # infs = np.diag(N * [-np.inf])
    # P_eps = np.exp(P+infs)

    # while np.any(np.sum(P_eps, axis=1) < min_thresh) \
    #         or np.any(np.sum(P_eps, axis=1) > max_thresh) \
    #         or np.any(np.sum(P_eps, axis=0) < min_thresh) \
    #         or np.any(np.sum(P_eps, axis=0) > max_thresh):
    for i in range(_max_iter):
        c = 1 / np.dot(P.T, r)
        r = 1 / np.dot(P, c)

        _D1 = np.diag(np.squeeze(r))
        _D2 = np.diag(np.squeeze(c))
        P_eps = np.dot(np.dot(_D1, P), _D2)

        # _iterations += 1

        if _iterations >= _max_iter:
            _stopping_condition = "max_iter"
            break

    # if not _stopping_condition:
    #     _stopping_condition = "epsilon"

    _D1 = np.diag(np.squeeze(r))
    _D2 = np.diag(np.squeeze(c))
    P_eps = np.dot(np.dot(_D1, P), _D2)

    return P_eps
