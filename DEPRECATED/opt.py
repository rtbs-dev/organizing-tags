from autograd import grad, value_and_grad
import autograd.numpy as np

from functools import partial
import DEPRECATED.loss as loss
import random

from tqdm.autonotebook import tqdm, trange

import warnings
warnings.warn("the `opt` module is deprecated!", DeprecationWarning,
              stacklevel=2)


def job_per_round(f, x0, obsv,
                  avg=True, decay=True,
                  callback=None, **kwargs):
    '''ASGD, requiring index of observation be passed to loss func'''
    x_avg = x0.copy() # running avg optimal
    x_hat = x0.copy() # per-round optimal

    reg = kwargs.get('reg', 1e-2)
    n_rep = kwargs.get('n_rep', 10) # gradient steps per observation
    γ0 = kwargs.get('learning_rate', 0.1)

    epochs = kwargs.get('epochs', 2)

    # for ADAM
    b1 = kwargs.get('b1', 0.9)
    b2 = kwargs.get('b2', 0.999)
    eps = kwargs.get('eps', 10 ** -8)

    η = γ0  # init
    m = np.zeros_like(x0)
    v = np.zeros_like(x0)
    μ = 1

    if callback is None:
        callback = lambda *args, **kws: None

    for epoch in range(epochs):
        samp = random.sample(obsv, k=len(obsv))
        for n, a in tqdm(enumerate(samp, 1), total=len(samp) - 1):
            n_iter = (epoch*len(obsv) + n)*n_rep
            for i in range(n_iter, n_iter + n_rep):
                f_inst = partial(f, obsv, n - 1)
                g = grad(f_inst)(x_hat, reg=reg)

                m = (1 - b1) * g + b1 * m  # First  moment estimate.
                v = (1 - b2) * (g ** 2) + b2 * v  # Second moment estimate.
                mhat = m / (1 - b1 ** (i + 1))  # Bias correction.
                vhat = v / (1 - b2 ** (i + 1))

                # exp. learning rate decay
                if decay:
                    η = γ0 * (1 + γ0 * reg * i / len(samp)) ** (-.75)

                # step w/ momentum
                x_hat = x_hat - η * mhat / (np.sqrt(vhat) + eps)
                # x_hat = x_hat - η * g  # no momentum

                # Averaging
                if avg:
                    μ = 1. / np.max([1., i - x_hat.size, i - len(obsv)])
                    x_avg = x_avg + μ * (x_hat - x_avg)
                else:
                    x_avg = x_hat

            P = loss._softmax(loss._symmetrize(x_avg), axis=1)
            callback(n_iter, P,
                     text=f'η={η:.2e}\nμ={μ:.2e}')


    return x_avg


def job_minibatch(f, x0, obsv,
                  avg=True, decay=True,
                  callback=None, **kwargs):
    """minibatch ASGD over observations"""

    def chunk(it, n):
        try:
            while True:
                xs = []  # The buffer to hold the next n items
                for _ in range(n):
                    xs.append(next(it))
                yield xs
        except StopIteration:
            yield xs

    x_avg = x0.copy()  # running avg optimal
    x_hat = x0.copy()  # per-round optimal

    sym = kwargs.get('sym', True)
    reg = kwargs.get('reg', 1e-2)
    epochs = kwargs.get('epochs', 10)
    batch_size = kwargs.get('batch_size', 5)
    γ0 = kwargs.get('learning_rate', 0.001)

    # for ADAM
    b1 = kwargs.get('b1', 0.9)
    b2 = kwargs.get('b2', 0.999)
    eps = kwargs.get('eps', 10 ** -8)

    η = γ0  # init
    m = np.zeros_like(x0)
    v = np.zeros_like(x0)
    μ = 1.

    if callback is None:
        callback = lambda *args, **kws: None

    for epoch in trange(1, epochs+1, desc='Epoch'):
        samp = random.sample(obsv, k=len(obsv))
        for n_iter, batch in tqdm(enumerate(chunk(iter(samp),
                                                  batch_size), 1),
                                  total=len(obsv)//batch_size, leave=False):
            i = epoch*len(obsv) + n_iter*batch_size
            f_inst = partial(f, batch, sym=sym)
            val, g = value_and_grad(f_inst)(x_hat, reg=reg)

            m = (1 - b1) * g + b1 * m  # First  moment estimate.
            v = (1 - b2) * (g ** 2) + b2 * v  # Second moment estimate.
            mhat = m / (1 - b1 ** (i + 1))  # Bias correction.
            vhat = v / (1 - b2 ** (i + 1))

            # exp. learning rate decay
            if decay:
                η = γ0 * (1 + γ0 * reg * i / len(samp)) ** (-.75)

            # step w/ momentum
            x_hat = x_hat - η * mhat / (np.sqrt(vhat) + eps)
            # x_hat = x_hat - η * g  # no momentum

            # Averaging
            avg_check = (i > x_hat.size) or (i > len(obsv))
            if avg and avg_check:
                μ = 1./np.max([1., i-x_hat.size, i-len(obsv)])
                x_avg = x_avg + μ * (x_hat - x_avg)
            else:
                μ = 0.
                x_avg = x_hat

            if sym:
                P = loss._softmax(loss._symmetrize(x_avg), axis=1)
            else:
                P = loss._softmax(x_avg, axis=1)
            # P = loss.sinkhorn(x_avg)
            callback(n_iter, P, val=val,
                     text=f'η={η:.2e}\nμ={μ:.2e}')
    # return loss._softmax(loss._symmetrize(x_avg), axis=1)
    return P
