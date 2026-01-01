import pytest

import torch
from torch import nn


@pytest.mark.parametrize("num_fracs", (1, 4))
@pytest.mark.parametrize("disable", (False, True))
def test_readme(num_fracs, disable):
    # a single branch layer

    branch = nn.Linear(512, 512)

    # before

    residual = torch.randn(2, 1024, 512)

    residual = branch(residual) + residual

    # after, say 4 streams in paper

    from hyper_connections import get_init_and_expand_reduce_stream_functions

    init_hyper_conn, expand_stream, reduce_stream = (
        get_init_and_expand_reduce_stream_functions(
            4, num_fracs=num_fracs, disable=disable
        )
    )

    # 1. wrap your branch function

    hyper_conn_branch = init_hyper_conn(dim=512, branch=branch)

    # 2. expand to 4 streams, this must be done before your trunk, typically a for-loop with many branch functions

    residual = expand_stream(residual)

    # 3. forward your residual as usual into the wrapped branch function(s)

    residual = hyper_conn_branch(residual)

    # 4. reduce 4 streams with a summation, this has to be done after your for-loop trunk. for transformer, unsure whether to do before or after final norm

    residual = reduce_stream(residual)

    assert residual.shape == (2, 1024, 512)


def test_manual():
    # a single branch layer

    branch = nn.Linear(512, 512)

    # before

    residual = torch.randn(2, 1024, 512)

    residual = branch(residual) + residual

    # after, say 4 streams in paper

    from hyper_connections import get_init_and_expand_reduce_stream_functions

    init_hyper_conn, expand_stream, reduce_stream = (
        get_init_and_expand_reduce_stream_functions(4)
    )

    # 1. instantiate hyper connection with correct number of streams (4 in this case) - or use the init function above

    hyper_conn = init_hyper_conn(dim=512)

    # 2. expand to 4 streams

    residual = expand_stream(residual)

    # 3. forward your residual into hyper connection for the branch input + add residual function (learned betas)

    branch_input, add_residual = hyper_conn(residual)

    branch_output = branch(branch_input)

    residual = add_residual(branch_output)

    # or you can do it in one line as so -> residual = hyper_conn.decorate_branch(branch)(residual)

    # 4. reduce 4 streams with a summation, this has to be done after your for loop trunk

    residual = reduce_stream(residual)
    assert residual.shape == (2, 1024, 512)


@pytest.mark.parametrize("disable", (False, True))
def test_multi_input_hyper_connections(disable):
    # two branch layers

    class CustomModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(512, 512)
            self.second_linear = nn.Linear(256, 512)
            self.third_linear = nn.Linear(128, 512)

        def forward(self, x, second, *, third):
            return self.linear(x) + self.second_linear(second) + self.third_linear(
                third
            ), 3.0

    branch = CustomModule()

    # before

    residual = torch.randn(3, 1024, 512)
    second_residual = torch.randn(3, 1024, 256)
    third_residual = torch.randn(3, 1024, 128)

    # residual = branch1(residual) + branch2(residual) + residual

    # after, say 4 streams in paper

    from hyper_connections.hyper_connections_with_multi_input_streams import (
        HyperConnections,
    )

    init_hyper_conn, expand_stream, reduce_stream = (
        HyperConnections.get_init_and_expand_reduce_stream_functions(4, disable=disable)
    )

    # 1. instantiate hyper connection with correct number of streams (4 in this case) - or use the init function above

    hyper_conn = init_hyper_conn(
        dim=512,
        branch=branch,
        additional_input_paths=[
            (1, 256),  # points at second residual stream, first arg
            ("third", 128),  # points at third residual stream, keyword argument 'third'
        ],
        layer_index=1,
    )

    # 2. expand to 4 streams

    residual = expand_stream(residual)
    second_residual = expand_stream(second_residual)
    third_residual = expand_stream(third_residual)

    # 3. forward your residual into hyper connection for the branch input + add residual function (learned betas)

    residual, rest_output = hyper_conn(residual, second_residual, third=third_residual)

    residual = reduce_stream(residual)

    assert residual.shape == (3, 1024, 512)


@pytest.mark.parametrize("disable", (False, True))
def test_residual_transform(disable):
    # a single branch layer

    branch = nn.Sequential(
        nn.Conv2d(512, 512, 3, padding=1), nn.SiLU(), nn.Conv2d(512, 256, 3, padding=1)
    )

    residual_fn = nn.Conv2d(512, 256, 1)

    # before

    residual = torch.randn(2, 512, 16, 16)

    before_residual = branch(residual) + residual_fn(residual)

    # after, say 4 streams in paper

    from hyper_connections import get_init_and_expand_reduce_stream_functions

    init_hyper_conn, expand_stream, reduce_stream = (
        get_init_and_expand_reduce_stream_functions(4, disable=disable)
    )

    # 1. wrap your branch function

    hyper_conn_branch = init_hyper_conn(
        dim=512, branch=branch, channel_first=True, residual_transform=residual_fn
    )

    # 2. expand to 4 streams, this must be done before your trunk, typically a for-loop with many branch functions

    residual = expand_stream(residual)

    # 3. forward your residual as usual into the wrapped branch function(s)

    residual = hyper_conn_branch(residual)

    # 4. reduce 4 streams with a summation, this has to be done after your for-loop trunk. for transformer, unsure whether to do before or after final norm

    after_residual = reduce_stream(residual)

    assert before_residual.shape == after_residual.shape


@pytest.mark.parametrize("disable", (False, True))
def test_channel_first_hyper_connection(disable):
    # a single branch layer

    branch = nn.Sequential(
        nn.Conv2d(512, 512, 3, padding=1), nn.SiLU(), nn.Conv2d(512, 256, 3, padding=1)
    )

    residual_fn = nn.Conv2d(512, 256, 1)

    # before

    residual = torch.randn(2, 512, 16, 16)

    before_residual = branch(residual) + residual_fn(residual)

    # after, say 4 streams in paper

    from hyper_connections.hyper_connections_channel_first import (
        get_init_and_expand_reduce_stream_functions,
    )

    init_hyper_conn, expand_stream, reduce_stream = (
        get_init_and_expand_reduce_stream_functions(4, disable=disable)
    )

    # 1. wrap your branch function

    hyper_conn_branch = init_hyper_conn(
        dim=512, branch=branch, residual_transform=residual_fn
    )

    # 2. expand to 4 streams, this must be done before your trunk, typically a for-loop with many branch functions

    residual = expand_stream(residual)

    # 3. forward your residual as usual into the wrapped branch function(s)

    residual = hyper_conn_branch(residual)

    # 4. reduce 4 streams with a summation, this has to be done after your for-loop trunk. for transformer, unsure whether to do before or after final norm

    after_residual = reduce_stream(residual)

    assert before_residual.shape == after_residual.shape
