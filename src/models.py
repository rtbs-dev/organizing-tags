__author__ = "Thurston Sexton"
"""
PyTorch re-implementation of the invite-based inference model.
"""

from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as f
# from src.analysis import Record
# from src.tools import minmax


class CensoredRW(torch.nn.Module):
    def __init__(self, N, reg=1e-2, sym=False, cuda=True):
        super(CensoredRW, self).__init__()

        self.device = torch.device(
            "cuda" if (torch.cuda.is_available() and cuda) else "cpu"
        )
        # self.to(self.device)
        self.N = N
        # self.P = torch.nn.Parameter(torch.Tensor(N, N).to(self.device))
        self.P = torch.nn.Parameter(torch.rand(N, N).float().to(self.device))

        self.i = (
            torch.ones(N, N)
            .triu().repeat(N, 1, 1)
            .permute(2, 0, 1)
        ).to(self.device).requires_grad_(False)
        self.j = (
            torch.ones(N, N)
            .triu().repeat(N, 1, 1)
            .permute(2, 1, 0)
        ).to(self.device).requires_grad_(False)

        self.reg = reg
        self.sym = sym

    def _symmetrize(self, a):  # assumes 0-diags

        bott = torch.tril(a) + torch.tril(a).t()
        top = torch.triu(a) + torch.triu(a).t()
        # return (bott+top)/2. + infs
        return torch.max(bott, top)  # - torch.diag(a)

    def _softmax(self, a, dim=1):
        a = a - a.max(dim=1, keepdim=True)[0]
        infs = torch.diagflat(
            torch.tensor(self.N*[-1*float('Inf')])
        ).to(self.device)

        a = torch.exp(a + infs)
        return f.normalize(a)

    def forward(self, M):
        # no_samp = len(M)
        def like(m):
            # A = self._softmax(torch.mm(self.P, self.P.t()))
            n = len(m)-1
            a = list(m) + list(set(range(self.N)) - set(m))
            index = torch.LongTensor(a).to(self.device)

            # t = A[index][:, index]
            # t = self._softmax(self._symmetrize(self.A))[index][:,index]
            if self.sym:
                t = f.normalize(
                    torch.exp(self._symmetrize(self.P[index][:, index])
                              ), p=1)
                # t = self._softmax(self._symmetrize(self.P))[index][:, index]
            else:
                t = f.normalize(torch.exp(self.P[index][:, index]), p=1)
            t = t - torch.diagflat(torch.diagonal(t))
            q_ma = (self.j*self.i)[:n]
            r_ma = ((1-self.i)*self.j)[:n]  # -\
            # (self.j*self.i)[n].flip(1)*self.j[:n]
            q = t*q_ma
            r = t*r_ma
            p = torch.matmul(
                torch.inverse(
                    torch.eye(self.N).to(self.device)-q
                ), r
            )
            lik = p.diagonal().diagonal(offset=-1).log().sum()

            return lik
        tr = tqdm(M, leave=False, desc='sample')
        loss = torch.stack([like(m) for m in tr])
        return -1*torch.sum(loss)


def train(model, x, epochs=50, batch_size=200,
          callback=False, compare_true=None, **opt_kws):
    # if torch.cuda.is_available():
    #     model.cuda()
    adam_kws = {'lr': 0.1}
    adam_kws.update(opt_kws)
    optimizer = torch.optim.Adam(model.parameters(), **adam_kws)
    if callback:
        fig, ax = plt.subplots(ncols=2)
        state = ax[1].imshow(np.random.rand(model.N, model.N))
        progress, = ax[0].plot(1)
        log = []
    # rec = Record(x_true=compare_true)
    # val = None
    for it in tqdm(range(epochs), desc='epoch'):

        optimizer.zero_grad()

        if batch_size is not None:
            for i in range(0, x.shape[0], batch_size):
                permutation = torch.randperm(x.shape[0])
                indices = permutation[i:i+batch_size]
                batch_x = x[indices]

                loss = model(batch_x)/batch_size
                loss.backward()
                optimizer.step()

        else:
            loss = model(x)
            loss.backward()
            optimizer.step()

        if callback:
            state.set_data(list(model.parameters())[0].cpu().data)
            log += [loss.item()]
            progress.set_data(np.arange(len(log)), log)
            ax[1].axis('off')
            ax[0].relim()
            ax[0].autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
            # print(val)

    return model
    # rec.err_and_mat(it, list(model.parameters())[0].data.numpy(), val=val)
