import numpy as np
import torch
from math import log, exp
import torch.distributed as dist

EPSILON = np.finfo(np.float32).tiny


class SubsetOperator(torch.nn.Module):
    # https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/sampling/subsets.html
    def __init__(self, k, hard=False):
        super(SubsetOperator, self).__init__()
        self.k = k
        self.hard = hard

    def forward(self, scores, tau):
        m = torch.distributions.gumbel.Gumbel(torch.zeros_like(scores), torch.ones_like(scores))
        g = m.sample()
        scores = scores + g

        # continuous top k
        khot = torch.zeros_like(scores)
        onehot_approx = torch.zeros_like(scores)
        for i in range(self.k):
            khot_mask = torch.max(1.0 - onehot_approx, torch.tensor([EPSILON]).cuda())
            scores = scores + torch.log(khot_mask)
            onehot_approx = torch.nn.functional.softmax(scores / tau, dim=1)
            khot = khot + onehot_approx

        if self.hard:
            # straight through
            khot_hard = torch.zeros_like(khot)
            val, ind = torch.topk(khot, self.k, dim=1)
            khot_hard = khot_hard.scatter_(1, ind, 1)
            res = khot_hard - khot.detach() + khot
        else:
            res = khot

        return res

def gumbel_topk(scores, k, tau, hard=False):

    if not dist.is_initialized():
        raise RuntimeError("Distributed process group is not initialized!")

    rank = dist.get_rank()
    if rank == 0:
        m = torch.distributions.gumbel.Gumbel(torch.zeros_like(scores), torch.ones_like(scores))
        g = m.sample()
    else:
        g = torch.empty_like(scores)  # Placeholder for other ranks

    # Synchronize the tensor across all ranks
    dist.broadcast(g, src=0)

    scores = scores + g

    # continuous top k
    khot = torch.zeros_like(scores)
    onehot_approx = torch.zeros_like(scores)
    for i in range(k):
        khot_mask = torch.max(1.0 - onehot_approx, torch.tensor([EPSILON]).to(scores.device))
        scores = scores + torch.log(khot_mask)
        onehot_approx = torch.nn.functional.softmax(scores / tau, dim=-1)
        khot = khot + onehot_approx

    if hard:
        # straight through
        khot_hard = torch.zeros_like(khot)
        val, ind = torch.topk(khot, k, dim=1)
        khot_hard = khot_hard.scatter_(1, ind, 1)
        res = khot_hard - khot.detach() + khot
    else:
        res = khot

    return res


# Differentiable subset pruning
def gumbel_soft_top_k(w, k, t, r=None):
    # https://github.com/rycolab/differentiable-subset-pruning/blob/master/transformers/src/transformers/gated_bert_utilities.py
    # https://github.com/rycolab/differentiable-subset-pruning/blob/master/transformers/src/transformers/modeling_gated_bert.py#L1055

    # apply gumbel noise
    if r is None:
        u = torch.rand_like(w) * (1 - EPSILON) + EPSILON
        r = -torch.log(-torch.log(u)) + w
    else:
        r = r + w
    epsilon = torch.ones_like(r)
    epsilon *= EPSILON

    # soft top k
    p = torch.zeros([k, w.size()[0]]).to(w.device).double()
    p[0] = torch.exp(torch.nn.functional.log_softmax(r / t, 0))
    for j in range(1, k):
        r += torch.log(torch.max(1 - p[j - 1], epsilon))
        p[j] = torch.exp(torch.nn.functional.log_softmax(r / t, 0))

    return p.sum(0)

class TauAnealing:

    def __init__(self, tau_ini, tau_end, n_cooldown):
        self.tau_ini = tau_ini
        self.tau_end = tau_end
        self.nc = n_cooldown

    def tau(self, n):
        log_tau = log(self.tau_ini) - min((n / self.nc), 1) * (log(self.tau_ini) - log(self.tau_end))
        return exp(log_tau)

def tau_aneal(tau_ini, tau_end, n, n_cooldown):
    # from "Differentiable Subset Pruning of Transformer Heads"
    log_tau = log(tau_ini) - min((n/n_cooldown),1) * (log(tau_ini) - log(tau_end))
    return exp(log_tau)
