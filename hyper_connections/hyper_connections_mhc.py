from typing import Callable

from functools import partial
from random import randrange
import math

import torch
from torch import nn
from torch.nn import Module
from torch.utils._pytree import tree_flatten, tree_unflatten

from einops import rearrange, einsum
from einops.layers.torch import Rearrange, Reduce

from hyper_connections.hyper_connections import Residual, StreamEmbed

"""
ein notation:
b - batch
d - feature dimension
s - residual streams
t - residual streams + num branch inputs
f - number of fractions (division of feature dimension space)
v - number of views for branch input
"""

# helper functions


def exists(v):
    return v is not None


def divisible_by(num, den):
    return (num % den) == 0


def default(v, d):
    return v if exists(v) else d


def add(x, y):
    return x + y


def sinkhorn_log(logits, num_iters=10, tau=0.05):
    n = logits.shape[-1]
    Z = logits / tau
    log_marginal = torch.zeros((n,), device=logits.device, dtype=logits.dtype)

    u = torch.zeros(logits.shape[:-1], device=Z.device, dtype=Z.dtype)
    v = torch.zeros_like(u)

    for _ in range(num_iters):
        u = log_marginal - torch.logsumexp(Z + v.unsqueeze(-2), dim=-1)
        v = log_marginal - torch.logsumexp(Z + u.unsqueeze(-1), dim=-2)

    return torch.exp(Z + u.unsqueeze(-1) + v.unsqueeze(-2))


# main functions


def get_expand_reduce_stream_functions(
    num_streams, add_stream_embed=False, dim=None, disable=False
):
    if num_streams == 1 or disable:
        return (nn.Identity(), nn.Identity())

    if add_stream_embed:
        assert exists(dim), (
            "`dim` must be passed into get_init_and_expand_reduce_stream_functions for returning an expansion function with stream embeddings added"
        )

        expand_fn = StreamEmbed(num_streams, dim, expand_to_streams=True)
    else:
        expand_fn = Reduce(
            pattern="b ... -> (b s) ...", reduction="repeat", s=num_streams
        )

    reduce_fn = Reduce(pattern="(b s) ... -> b ...", reduction="sum", s=num_streams)

    return expand_fn, reduce_fn


def get_init_and_expand_reduce_stream_functions(
    num_streams,
    num_fracs=1,
    dim=None,
    add_stream_embed=False,
    disable=None,
):
    disable = default(disable, num_streams == 1 and num_fracs == 1)

    hyper_conn_klass = HyperConnections if not disable else Residual

    init_hyper_conn_fn = partial(hyper_conn_klass, num_streams, num_fracs=num_fracs)
    expand_reduce_fns = get_expand_reduce_stream_functions(
        num_streams, add_stream_embed=add_stream_embed, dim=dim, disable=disable
    )

    if exists(dim):
        init_hyper_conn_fn = partial(init_hyper_conn_fn, dim=dim)

    return (init_hyper_conn_fn, *expand_reduce_fns)


# main classes


class HyperConnections(Module):
    def __init__(
        self,
        num_residual_streams,
        *,
        dim,
        branch: Module | None = None,
        layer_index=None,
        channel_first=False,
        dropout=0.0,
        residual_transform: Module
        | None = None,  # to support resnet blocks where dimension in not equal to dimension out - usually a residual conv
        add_branch_out_to_residual=True,  # will disable depth connections (weighted residual sum with beta) if set False
        num_input_views=1,  # allow for the branch module to receive multiple input views, dimension placed on the very left (before batch)
        depth_residual_fn=add,
        num_fracs=1,  # https://arxiv.org/abs/2503.14125
        mhc_num_iters=10,
        mhc_tau=0.05,
        mhc_residual_identity_mix=False,
        mhc_residual_alpha=0.01,
    ):
        """
        Appendix J, Algorithm2 in - https://arxiv.org/abs/2409.19606
        """
        super().__init__()

        self.branch = branch
        self.mhc_num_iters = mhc_num_iters
        self.mhc_tau = mhc_tau
        self.mhc_residual_identity_mix = mhc_residual_identity_mix

        # frac-connections paper - num_fracs > 1 will be the `m` in their paper https://arxiv.org/abs/2503.14125

        assert num_fracs >= 1
        assert num_fracs == 1, "`num_fracs` must be 1 for mHC"

        self.num_fracs = num_fracs
        self.has_fracs = num_fracs > 1

        self.split_fracs = Rearrange("b ... (f d) -> b ... f d", f=num_fracs)
        self.merge_fracs = Rearrange("b ... f d -> b ... (f d)")

        assert divisible_by(dim, num_fracs), (
            f"feature dimension ({dim}) must be divisible by the `num_fracs` ({num_fracs})"
        )

        dim //= num_fracs  # effective dim handled in dimension is feature dimension divided by num fractions

        assert num_residual_streams > 0, "`num_residual_streams` must be greater than 0"

        self.num_residual_streams = num_residual_streams
        init_residual_index = (
            default(layer_index, randrange(num_residual_streams)) % num_residual_streams
        )  # just choose one random residual stream if layer index not given

        # width num residual streams

        assert num_input_views >= 1
        assert num_input_views == 1, "`num_input_views` must be 1 for mHC"
        self.num_input_views = num_input_views

        self.add_branch_out_to_residual = add_branch_out_to_residual

        # width connection

        init_h_res = torch.full((num_residual_streams, num_residual_streams), -8.0)
        init_h_res.fill_diagonal_(0.0)
        self.H_res_logits = nn.Parameter(init_h_res)

        init_h_pre = torch.full((num_input_views, num_residual_streams), -8.0)
        init_h_pre[:, init_residual_index] = 0.0
        self.H_pre_logits = nn.Parameter(init_h_pre)

        if add_branch_out_to_residual:
            self.H_post_logits = nn.Parameter(
                torch.zeros(num_input_views, num_residual_streams)
            )

        if mhc_residual_identity_mix:
            alpha_clamped = max(1e-4, min(1 - 1e-4, mhc_residual_alpha))
            alpha_logit_init = math.log(alpha_clamped / (1 - alpha_clamped))
            self.H_res_alpha_logit = nn.Parameter(torch.tensor(alpha_logit_init))

        # dropouts

        self.dropout = nn.Dropout(dropout)

        # channel first option

        self.channel_first = channel_first

        # maybe residual transform

        self.residual_transform = default(residual_transform, nn.Identity())

        # maybe custom depth connection residual function
        # this is to prepare for gating the addition of the branch outputs to the residual streams
        # needed for memory lanes a la RMT / LMM

        self.depth_residual_fn = depth_residual_fn

    def width_connection(self, residuals):
        streams = self.num_residual_streams

        maybe_transformed_residuals = self.residual_transform(residuals)

        # width connection

        if self.channel_first:
            residuals = rearrange(residuals, "b d ... -> b ... d")
            maybe_transformed_residuals = rearrange(
                maybe_transformed_residuals, "b d ... -> b ... d"
            )

        residuals = self.split_fracs(residuals)
        maybe_transformed_residuals = self.split_fracs(maybe_transformed_residuals)

        residuals = rearrange(residuals, "(b s) ... d -> b ... s d", s=streams)
        maybe_transformed_residuals = rearrange(
            maybe_transformed_residuals, "(b s) ... d -> b ... s d", s=streams
        )

        S = sinkhorn_log(
            self.H_res_logits, num_iters=self.mhc_num_iters, tau=self.mhc_tau
        )

        if self.mhc_residual_identity_mix:
            alpha = torch.sigmoid(self.H_res_alpha_logit)
            I = torch.eye(streams, device=S.device, dtype=S.dtype)
            h_res = (1 - alpha) * I + alpha * S
        else:
            h_res = S

        residuals_out = einsum(
            h_res, maybe_transformed_residuals, "s t, ... s d -> ... t d"
        )

        h_pre = self.H_pre_logits.softmax(dim=-1)
        branch_input = einsum(h_pre, residuals, "v s, ... s d -> ... v d")

        h_post = None
        if self.add_branch_out_to_residual:
            h_post = self.H_post_logits.softmax(dim=-1)

        if getattr(self, "collect_stats", False):
            with torch.no_grad():
                stats = dict(
                    h_res_min=h_res.min(),
                    h_res_row_sum=h_res.sum(dim=-1).mean(),
                    h_res_col_sum=h_res.sum(dim=-2).mean(),
                    h_pre_min=h_pre.min(),
                )
                if h_post is not None:
                    stats["h_post_min"] = h_post.min()
                if self.mhc_residual_identity_mix:
                    stats["h_res_alpha"] = torch.sigmoid(self.H_res_alpha_logit)
                self.last_stats = {k: v.detach() for k, v in stats.items()}

        if self.num_input_views == 1:
            branch_input = branch_input[..., 0, :]
        else:
            branch_input = rearrange(branch_input, "b ... v d -> v b ... d")

        if self.channel_first:
            branch_input = rearrange(branch_input, "b ... d -> b d ...")

        branch_input = self.merge_fracs(branch_input)

        residuals_out = rearrange(residuals_out, "b ... s d -> (b s) ... d")
        residuals_out = self.merge_fracs(residuals_out)

        if self.channel_first:
            residuals_out = rearrange(residuals_out, "b ... d -> b d ...")

        return branch_input, residuals_out, dict(beta=h_post)

    def depth_connection(self, branch_output, residuals, *, beta):
        assert self.add_branch_out_to_residual
        assert beta is not None

        branch_output = self.split_fracs(branch_output)

        if self.channel_first:
            branch_output = rearrange(branch_output, "b d ... -> b ... d")

        if beta.ndim == 2:
            beta = beta[0]

        output = einsum(branch_output, beta, "b ... d, s -> b ... s d")
        output = rearrange(output, "b ... s d -> (b s) ... d")

        output = self.merge_fracs(output)

        if self.channel_first:
            output = rearrange(output, "b ... d -> b d ...")

        residuals = self.depth_residual_fn(output, residuals)

        return self.dropout(residuals)

    def decorate_branch(self, branch: Callable):
        assert not exists(self.branch), "branch was already wrapped on init"

        def forward_and_add_residual(residual, *args, **kwargs):
            branch_input, add_residual = self.forward(residual)

            branch_output = branch(branch_input, *args, **kwargs)

            residual = add_residual(branch_output)

            return residual

        return forward_and_add_residual

    def forward(self, residuals, *branch_args, **branch_kwargs):
        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):
            if not self.add_branch_out_to_residual:
                return branch_out

            (branch_out, *rest), tree_spec = tree_flatten(branch_out)

            branch_out = self.depth_connection(branch_out, residuals, **residual_kwargs)

            return tree_unflatten((branch_out, *rest), tree_spec)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)

        return add_residual_fn(branch_output)


HyperConnections.get_expand_reduce_stream_functions = staticmethod(
    get_expand_reduce_stream_functions
)
HyperConnections.get_init_and_expand_reduce_stream_functions = staticmethod(
    get_init_and_expand_reduce_stream_functions
)
