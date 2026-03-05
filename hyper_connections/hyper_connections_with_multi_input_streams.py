from __future__ import annotations
from typing import Callable, Union

from functools import partial
from random import randrange

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.utils._pytree import tree_flatten, tree_unflatten

from einops import rearrange, repeat, reduce, einsum
from einops.layers.torch import Rearrange, Reduce

"""
ein notation:
b - batch
d - feature dimension
s - residual streams
t - residual streams + num branch inputs
"""

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

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

# norms

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * (self.gamma + 1)

class ProjActScale(Module):
    def __init__(
        self,
        dim,
        dim_out,
        activation: Module = nn.Identity(),
        scale_init: float = 1e-2,
        squeeze_output = False
    ):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.proj = nn.Linear(dim, dim_out, bias = False)
        nn.init.zeros_(self.proj.weight)

        self.act = activation
        self.scale = nn.Parameter(torch.ones(()) * scale_init)
        self.maybe_squeeze = Rearrange('... 1 -> ...') if squeeze_output else nn.Identity()

    def forward(self, x):
        out = self.proj(x)
        out = self.act(out)
        return self.maybe_squeeze(out * self.scale)

# main classes

# residual base class

class Residual(Module):
    def __init__(
        self,
        *args,
        branch=None,
        **kwargs
    ):
        super().__init__()
        self.branch = branch

    def width_connection(self, residuals, *args, **kwargs):
        return residuals, residuals, dict()

    def depth_connection(self, branch_output, residuals):
        return branch_output + residuals

    def decorate_branch(self, branch: Callable):
        assert not exists(self.branch), 'branch was already wrapped on init'

        def forward_and_add_residual(residual, *args, **kwargs):
            branch_input, add_residual = self.forward(residual, *args, **kwargs)

            branch_output = branch(branch_input, *args, **kwargs)

            residual = add_residual(branch_output)

            return residual

        return forward_and_add_residual

    def forward(self, residuals, *branch_args, **branch_kwargs):

        branch_input, residuals, residual_kwargs = self.width_connection(residuals, *branch_args, **branch_kwargs)

        def add_residual_fn(branch_out):
            (branch_out, *rest), tree_spec = tree_flatten(branch_out)

            branch_out = self.depth_connection(branch_out, residuals, **residual_kwargs)

            return tree_unflatten((branch_out, *rest), tree_spec)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)

        return add_residual_fn(branch_output)

# hyper connection with multiple input streams

InputPathType = Union[int, str]  # the path to the second residual stream, where `int` points to *args[`int`] and `str` points to **kwargs[`str`] - `int` needs to be > 0, as 0 is the default input residual stream

class HyperConnections(Module):
    def __init__(
        self,
        num_residual_streams,
        *,
        dim,
        additional_input_paths = None, # if an additional residual has different dimensions, pass (path, dim)
        branch=None,
        layer_index = None,
        tanh = True,
        channel_first = False,
        dropout = 0.
    ):
        """
        Appendix J, Algorithm2 in - https://arxiv.org/abs/2409.19606
        """
        super().__init__()

        self.branch = branch
        act = nn.Tanh() if tanh else nn.Identity()

        self.num_residual_streams = num_residual_streams
        assert num_residual_streams > 0, '`num_residual_streams` must be greater than 0'

        # activation, seemingly results were wishy washy depending on using tanh or not

        self.norm = RMSNorm(dim) # they used layernorm in paper, but rmsnorm is fine given what we know now

        init_residual_index = default(layer_index, randrange(num_residual_streams)) % num_residual_streams # just choose one random residual stream if layer index not given

        init_alpha0 = torch.zeros((num_residual_streams, 1))
        init_alpha0[init_residual_index, 0] = 1.

        self.dynamic_alpha_and_branch_input = ProjActScale(dim, num_residual_streams + 1, activation = act)
        self.static_alpha = nn.Parameter(torch.cat([init_alpha0, torch.eye(num_residual_streams)], dim = 1))

        self.dynamic_beta = ProjActScale(dim, 1, activation = act, squeeze_output = True)
        self.static_beta = nn.Parameter(torch.ones(num_residual_streams))

        # additional input residual streams

        additional_input_paths = default(additional_input_paths, [])
        additional_input_paths = [one_path if isinstance(one_path, tuple) else (one_path, dim) for one_path in additional_input_paths]

        assert all([isinstance(path, str) or path > 0 for (path, _) in additional_input_paths])

        self.additional_norms = ModuleList([RMSNorm(dim) for _, dim in additional_input_paths])
        self.additional_to_dynamic_input = ModuleList([ProjActScale(dim, 1, activation = act, squeeze_output = True) for _ , dim in additional_input_paths])
        self.additional_static_input = nn.ParameterList([nn.Parameter(init_alpha0[..., 0]) for _ in additional_input_paths])

        self.additional_input_paths = additional_input_paths

        # dropouts

        self.dropout = nn.Dropout(dropout)

        # channel first option

        self.channel_first = channel_first

    def width_connection(
        self,
        residuals,
        *branch_args,
        **branch_kwargs
    ):

        transpose = self.channel_first

        # width connection

        if transpose:
            residuals = rearrange(residuals, 'b d ... -> b ... d')

        residuals = rearrange(residuals, '(b s) ... d -> b ... s d', s = self.num_residual_streams)

        normed = self.norm(residuals)

        # alpha for weighted sum of residuals going into branch

        dynamic_alpha = self.dynamic_alpha_and_branch_input(normed)
        alpha = dynamic_alpha + self.static_alpha

        # beta for weights from branch output back to residual streams

        dynamic_beta = self.dynamic_beta(normed)
        beta = dynamic_beta + self.static_beta

        mix_h = einsum(alpha, residuals, '... s t, ... s d -> ... t d')

        branch_input, residuals = mix_h[..., 0, :], mix_h[..., 1:, :]

        if transpose:
            branch_input = rearrange(branch_input, 'b ... d -> b d ...')

        # take care of additional inputs

        branch_args = list(branch_args)

        for (path, *_), norm, proj, learned_static in zip(self.additional_input_paths, self.additional_norms, self.additional_to_dynamic_input, self.additional_static_input):

            # get the residual streams from additional arguments

            if isinstance(path, int):
                additional_residuals = branch_args[path - 1]
            elif isinstance(path, str):
                additional_residuals = branch_kwargs[path]

            assert torch.is_tensor(additional_residuals)

            # handle channel first

            if transpose:
                # FIX: pass `additional_residuals` as the first arg to einops.rearrange (previously missing and would error)
                additional_residuals = rearrange(additional_residuals, 'b d ... -> b ... d')

            additional_residuals = rearrange(additional_residuals, '(b s) ... d -> b ... s d', s = self.num_residual_streams)

            # norm

            additional_mix = proj(norm(additional_residuals))
            additional_mix = additional_mix + learned_static

            additional_residuals = einsum(additional_mix, additional_residuals, '... s, ... s d -> ... d')

            # transpose out

            if transpose:
                # FIX: pass `additional_residuals` as the first arg to einops.rearrange (previously missing and would error)
                additional_residuals = rearrange(additional_residuals, 'b ... d -> b d ...')

            # set back transformed residual

            if isinstance(path, int):
                branch_args[path - 1] = additional_residuals
            elif isinstance(path, str):
                branch_kwargs[path] = additional_residuals

        return ([branch_input, *branch_args], branch_kwargs), residuals, dict(beta = beta)

    def depth_connection(self, branch_output, residuals, *, beta):
        # 'depth' connection

        if self.channel_first:
            branch_output = rearrange(branch_output, 'b d ... -> b ... d')

        residuals = einsum(branch_output, beta, 'b ... d, b ... s -> b ... s d') + residuals
        output = rearrange(residuals, 'b ... s d -> (b s) ... d')

        if self.channel_first:
            output = rearrange(output, 'b ... d -> b d ...')

        return self.dropout(output)

    def decorate_branch(self, branch: Callable):
        assert not exists(self.branch), 'branch was already wrapped on init'

        def forward_and_add_residual(residual, *args, **kwargs):
            ([branch_input, *args], kwargs), add_residual = self.forward(residual, *args, **kwargs)

            branch_output = branch(branch_input, *args, **kwargs)

            residual = add_residual(branch_output)

            return residual

        return forward_and_add_residual

    def forward(self, residuals, *branch_args, **branch_kwargs):

        (branch_args, branch_kwargs), residuals, residual_kwargs = self.width_connection(residuals, *branch_args, **branch_kwargs)

        def add_residual_fn(branch_out):
            (branch_out, *rest), tree_spec = tree_flatten(branch_out)

            branch_out = self.depth_connection(branch_out, residuals, **residual_kwargs)

            return tree_unflatten((branch_out, *rest), tree_spec)

        if not exists(self.branch):
            return (branch_args, branch_kwargs), add_residual_fn

        branch_output = self.branch(*branch_args, **branch_kwargs)

        return add_residual_fn(branch_output)

# add static methods

HyperConnections.get_expand_reduce_stream_functions = staticmethod(get_expand_reduce_stream_functions)
HyperConnections.get_init_and_expand_reduce_stream_functions = staticmethod(get_init_and_expand_reduce_stream_functions)
