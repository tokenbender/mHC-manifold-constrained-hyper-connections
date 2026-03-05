from __future__ import annotations
from typing import Callable

from functools import partial
from random import randrange

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.utils._pytree import tree_flatten, tree_unflatten

from einops import rearrange, repeat, reduce, einsum
from einops.layers.torch import Reduce

"""
ein notation:
b - batch
d - feature dimension
s - residual streams
i - branch inputs
br - branch functions
t - residual streams + num branch inputs
"""

from hyper_connections.hyper_connections import Residual, StreamEmbed, RMSNorm

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def identity(t):
    return t

# main functions

def get_expand_reduce_stream_functions(num_streams, disable = False):
    if num_streams == 1 or disable:
        return (nn.Identity(), nn.Identity())

    expand_fn = Reduce(pattern = 'b ... -> (b s) ...', reduction = 'repeat', s = num_streams)
    reduce_fn = Reduce(pattern = '(b s) ... -> b ...', reduction = 'sum', s = num_streams)

    return expand_fn, reduce_fn

def get_init_and_expand_reduce_stream_functions(num_streams, disable = None):

    disable = default(disable, num_streams == 1)

    hyper_conn_klass = HyperConnections if not disable else Residual

    init_hyper_conn_fn = partial(hyper_conn_klass, num_streams)
    expand_reduce_fns = get_expand_reduce_stream_functions(num_streams, disable = disable)

    return (init_hyper_conn_fn, *expand_reduce_fns)

# main classes

# hyper connection residual streams

class HyperConnections(Module):
    def __init__(
        self,
        num_residual_streams,
        *,
        dim,
        branch=None,
        layer_index = None,
        tanh = True,
        channel_first = False,
        dropout = 0.,
        num_branch_inputs = 1  # residuals will be linearly combined to multiple inputs, fed through the branch, then linearly combined back out to residuals
    ):
        """
        Appendix J, Algorithm2 in - https://arxiv.org/abs/2409.19606
        """
        super().__init__()

        self.branches = None

        if isinstance(branch, Module):
            branch = [branch]

        if exists(branch):
            assert divisible_by(num_branch_inputs, len(branch))

            self.branches = ModuleList(branch)

        # activation, seemingly results were wishy washy depending on using tanh or not

        self.act = nn.Tanh() if tanh else nn.Identity()

        self.norm = RMSNorm(dim) # they used layernorm in paper, but rmsnorm is fine given what we know now

        self.num_residual_streams = num_residual_streams
        self.num_branch_inputs = num_branch_inputs

        self.static_beta = nn.Parameter(torch.ones(num_residual_streams, num_branch_inputs))

        # make sure each branch input receives from different residual stream on init

        stream_branches = num_residual_streams * num_branch_inputs
        layer_index = default(layer_index, randrange(stream_branches))
        # FIX: offset should cycle through residual streams;
        #  previous formula used `stream_branches * num_branch_inputs` and caused repeated/biased init patterns
        layer_offset = layer_index % num_residual_streams

        stream_seq = torch.arange(num_residual_streams)
        branch_input_seq = torch.arange(num_branch_inputs)

        init_alpha0 = rearrange(stream_seq, 's -> s 1') + rearrange(branch_input_seq, 'bi -> 1 bi') + layer_offset
        init_alpha0 = ((init_alpha0 % num_residual_streams) == 0).float()

        self.static_alpha = nn.Parameter(torch.cat([init_alpha0, torch.eye(num_residual_streams)], dim = 1))

        self.dynamic_alpha_fn = nn.Parameter(torch.zeros(dim, num_residual_streams + num_branch_inputs))
        self.dynamic_alpha_scale = nn.Parameter(torch.ones(()) * 1e-2)
        self.dynamic_beta_fn = nn.Parameter(torch.zeros(dim, num_branch_inputs))
        self.dynamic_beta_scale = nn.Parameter(torch.ones(()) * 1e-2)

        # dropout

        self.dropout = nn.Dropout(dropout)

        # channel first option

        self.channel_first = channel_first

    def width_connection(self, residuals):
        num_streams, num_branch_inputs = self.num_residual_streams, self.num_branch_inputs

        # width connection

        if self.channel_first:
            residuals = rearrange(residuals, 'b d ... -> b ... d')

        residuals = rearrange(residuals, '(b s) ... d -> b ... s d', s = num_streams)

        normed = self.norm(residuals)

        # alpha for weighted sum of residuals going into branch

        wc_weight = self.act(normed @ self.dynamic_alpha_fn)
        dynamic_alpha = wc_weight * self.dynamic_alpha_scale
        alpha = dynamic_alpha + self.static_alpha

        # beta for weights from branch output back to residual streams

        dc_weight = self.act(normed @ self.dynamic_beta_fn)
        dynamic_beta = dc_weight * self.dynamic_beta_scale

        beta = dynamic_beta + self.static_beta

        mix_h = einsum(alpha, residuals, '... s t, ... s d -> ... t d')

        branch_input, residuals = mix_h[..., :-num_streams, :], mix_h[..., -num_streams:, :]

        branch_input = rearrange(branch_input, 'b ... i d -> (i b) ... d')

        if self.channel_first:
            branch_input = rearrange(branch_input, 'b ... d -> b d ...')

        return branch_input, residuals, dict(beta = beta)

    def depth_connection(self, branch_output, residuals, *, beta):
        # 'depth' connection

        if self.channel_first:
            branch_output = rearrange(branch_output, 'b d ... -> b ... d')

        branch_output = rearrange(branch_output, '(i b) ... -> i b ...', i = self.num_branch_inputs)

        residuals = einsum(branch_output, beta, 'i b ... d, b ... s i -> b ... s d') + residuals

        output = rearrange(residuals, 'b ... s d -> (b s) ... d')

        if self.channel_first:
            output = rearrange(output, 'b ... d -> b d ...')

        return self.dropout(output)

    def decorate_branch(self, branch):
        assert not exists(self.branches), 'branch was already wrapped on init'

        def forward_and_add_residual(residual, *args, **kwargs):
            branch_input, add_residual = self.forward(residual)

            if callable(branch):
                branches = [branch]
            else:
                branches = branch

            branch_inputs = rearrange(branch_input, '(br b) ... -> br b ...', br = len(branches))

            branch_outputs = [fn(x, *args, **kwargs) for fn, x in zip(branches, branch_inputs)]

            branch_output = torch.cat(branch_outputs)

            residual = add_residual(branch_output)

            return residual

        return forward_and_add_residual

    def forward(self, residuals, *branch_args, **branch_kwargs):

        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):
            (branch_out, *rest), tree_spec = tree_flatten(branch_out)

            branch_out = self.depth_connection(branch_out, residuals, **residual_kwargs)

            return tree_unflatten((branch_out, *rest), tree_spec)

        if not exists(self.branches):
            return branch_input, add_residual_fn

        branch_inputs = rearrange(branch_input, '(br b) ... -> br b ...', br = len(self.branches))

        branch_outputs = [fn(x, *branch_args, **branch_kwargs) for fn, x in zip(self.branches, branch_inputs)]

        branch_output = torch.cat(branch_outputs)

        return add_residual_fn(branch_output)

HyperConnections.get_expand_reduce_stream_functions = staticmethod(get_expand_reduce_stream_functions)
HyperConnections.get_init_and_expand_reduce_stream_functions = staticmethod(get_init_and_expand_reduce_stream_functions)
