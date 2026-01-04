import pytest

import torch
from torch import nn


@pytest.mark.parametrize("num_fracs", (1, 4))
@pytest.mark.parametrize("disable", (False, True))
def test_readme(num_fracs, disable):
    from hyper_connections import get_init_and_expand_reduce_stream_functions

    branch = nn.Linear(512, 512)
    residual = torch.randn(2, 1024, 512)

    init_hyper_conn, expand_stream, reduce_stream = (
        get_init_and_expand_reduce_stream_functions(
            4, num_fracs=num_fracs, disable=disable
        )
    )

    hyper_conn_branch = init_hyper_conn(dim=512, branch=branch)

    residual = expand_stream(residual)
    residual = hyper_conn_branch(residual)
    residual = reduce_stream(residual)

    assert residual.shape == (2, 1024, 512)


def test_manual():
    from hyper_connections import get_init_and_expand_reduce_stream_functions

    branch = nn.Linear(512, 512)
    residual = torch.randn(2, 1024, 512)

    init_hyper_conn, expand_stream, reduce_stream = (
        get_init_and_expand_reduce_stream_functions(4)
    )

    hyper_conn = init_hyper_conn(dim=512)

    residual = expand_stream(residual)
    branch_input, add_residual = hyper_conn(residual)
    branch_output = branch(branch_input)
    residual = add_residual(branch_output)
    residual = reduce_stream(residual)

    assert residual.shape == (2, 1024, 512)


@pytest.mark.parametrize("disable", (False, True))
def test_multi_input_hyper_connections(disable):
    from hyper_connections.hyper_connections_with_multi_input_streams import (
        HyperConnections,
    )

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

    residual = torch.randn(3, 1024, 512)
    second_residual = torch.randn(3, 1024, 256)
    third_residual = torch.randn(3, 1024, 128)

    init_hyper_conn, expand_stream, reduce_stream = (
        HyperConnections.get_init_and_expand_reduce_stream_functions(4, disable=disable)
    )

    hyper_conn = init_hyper_conn(
        dim=512,
        branch=branch,
        additional_input_paths=[
            (1, 256),
            ("third", 128),
        ],
        layer_index=1,
    )

    residual = expand_stream(residual)
    second_residual = expand_stream(second_residual)
    third_residual = expand_stream(third_residual)

    residual, rest_output = hyper_conn(residual, second_residual, third=third_residual)
    residual = reduce_stream(residual)

    assert residual.shape == (3, 1024, 512)


@pytest.mark.parametrize("disable", (False, True))
def test_residual_transform(disable):
    from hyper_connections import get_init_and_expand_reduce_stream_functions

    branch = nn.Sequential(
        nn.Conv2d(512, 512, 3, padding=1), nn.SiLU(), nn.Conv2d(512, 256, 3, padding=1)
    )
    residual_fn = nn.Conv2d(512, 256, 1)

    residual = torch.randn(2, 512, 16, 16)
    before_residual = branch(residual) + residual_fn(residual)

    init_hyper_conn, expand_stream, reduce_stream = (
        get_init_and_expand_reduce_stream_functions(4, disable=disable)
    )

    hyper_conn_branch = init_hyper_conn(
        dim=512, branch=branch, channel_first=True, residual_transform=residual_fn
    )

    residual = expand_stream(residual)
    residual = hyper_conn_branch(residual)
    after_residual = reduce_stream(residual)

    assert before_residual.shape == after_residual.shape


@pytest.mark.parametrize("disable", (False, True))
def test_channel_first_hyper_connection(disable):
    from hyper_connections.hyper_connections_channel_first import (
        get_init_and_expand_reduce_stream_functions,
    )

    branch = nn.Sequential(
        nn.Conv2d(512, 512, 3, padding=1), nn.SiLU(), nn.Conv2d(512, 256, 3, padding=1)
    )
    residual_fn = nn.Conv2d(512, 256, 1)

    residual = torch.randn(2, 512, 16, 16)
    before_residual = branch(residual) + residual_fn(residual)

    init_hyper_conn, expand_stream, reduce_stream = (
        get_init_and_expand_reduce_stream_functions(4, disable=disable)
    )

    hyper_conn_branch = init_hyper_conn(
        dim=512, branch=branch, residual_transform=residual_fn
    )

    residual = expand_stream(residual)
    residual = hyper_conn_branch(residual)
    after_residual = reduce_stream(residual)

    assert before_residual.shape == after_residual.shape


def test_disable_matches_residual():
    from hyper_connections import get_init_and_expand_reduce_stream_functions

    torch.manual_seed(0)

    branch = nn.Linear(32, 32)
    residual = torch.randn(2, 16, 32)
    expected = branch(residual) + residual

    init_hyper_conn, expand_stream, reduce_stream = (
        get_init_and_expand_reduce_stream_functions(4, disable=True)
    )

    hyper_conn_branch = init_hyper_conn(dim=32, branch=branch)
    output = reduce_stream(hyper_conn_branch(expand_stream(residual)))

    torch.testing.assert_close(output, expected)


def test_decorate_matches_manual():
    from hyper_connections import get_init_and_expand_reduce_stream_functions

    torch.manual_seed(0)

    branch = nn.Linear(32, 32)
    residual = torch.randn(2, 16, 32)

    init_hyper_conn, expand_stream, reduce_stream = (
        get_init_and_expand_reduce_stream_functions(4)
    )

    hyper_conn = init_hyper_conn(dim=32)
    hyper_conn_branch = hyper_conn.decorate_branch(branch)

    expanded = expand_stream(residual)
    output_decorated = reduce_stream(hyper_conn_branch(expanded))

    branch_input, add_residual = hyper_conn(expanded)
    output_manual = reduce_stream(add_residual(branch(branch_input)))

    torch.testing.assert_close(output_decorated, output_manual)


def test_backward_smoke():
    from hyper_connections import get_init_and_expand_reduce_stream_functions

    torch.manual_seed(0)

    branch = nn.Linear(16, 16)
    residual = torch.randn(2, 8, 16, requires_grad=True)

    init_hyper_conn, expand_stream, reduce_stream = (
        get_init_and_expand_reduce_stream_functions(4)
    )

    hyper_conn_branch = init_hyper_conn(dim=16, branch=branch)

    output = reduce_stream(hyper_conn_branch(expand_stream(residual)))
    loss = output.sum()
    loss.backward()

    assert residual.grad is not None


def test_mhc_H_res_constraints():
    from hyper_connections.hyper_connections import HyperConnections, sinkhorn_log

    hc = HyperConnections(num_residual_streams=4, dim=64, mhc=True)
    H_res = sinkhorn_log(hc.H_res_logits, hc.sinkhorn_iters, hc.sinkhorn_tau)

    assert H_res.min().item() >= 0
    assert torch.allclose(
        H_res.sum(dim=-1),
        torch.ones(4, device=H_res.device, dtype=H_res.dtype),
        atol=1e-3,
    )
    assert torch.allclose(
        H_res.sum(dim=-2),
        torch.ones(4, device=H_res.device, dtype=H_res.dtype),
        atol=1e-3,
    )


def test_mhc_orthostochastic_constraints():
    from hyper_connections.hyper_connections import (
        HyperConnections,
        orthostochastic_project,
        zeropower_via_newtonschulz,
    )

    hc = HyperConnections(
        num_residual_streams=4,
        dim=64,
        mhc=True,
        mhc_h_res_proj="orthostochastic",
    )
    O = zeropower_via_newtonschulz(
        hc.H_res_logits,
        steps=hc.ns_steps,
        eps=hc.ns_eps,
        coeffs=hc.ns_coeffs,
    )
    H_res = orthostochastic_project(
        hc.H_res_logits,
        ns_steps=hc.ns_steps,
        ns_eps=hc.ns_eps,
        ns_coeffs=hc.ns_coeffs,
    )

    assert H_res.min().item() >= 0
    assert torch.allclose(
        O @ O.T,
        torch.eye(4, device=O.device, dtype=O.dtype),
        atol=1e-2,
    )


def test_mhc_H_pre_H_post_constraints():
    from hyper_connections.hyper_connections import HyperConnections

    hc = HyperConnections(num_residual_streams=4, dim=64, mhc=True)
    H_pre = torch.softmax(hc.H_pre_logits, dim=-1)
    H_post = torch.softmax(hc.H_post_logits, dim=-1)

    assert H_pre.min().item() >= 0
    assert H_post.min().item() >= 0
    assert torch.allclose(
        H_pre.sum(),
        torch.ones((), device=H_pre.device, dtype=H_pre.dtype),
        atol=1e-6,
    )
    assert torch.allclose(
        H_post.sum(),
        torch.ones((), device=H_post.device, dtype=H_post.dtype),
        atol=1e-6,
    )


def test_mhc_forward_shapes():
    from hyper_connections.hyper_connections import HyperConnections

    streams, dim, batch, seq = 4, 64, 2, 8
    hc = HyperConnections(num_residual_streams=streams, dim=dim, mhc=True)
    x = torch.randn(batch * streams, seq, dim)

    branch_input, add_residual = hc(x)
    assert branch_input.shape == (batch, seq, dim)

    branch_output = torch.randn(batch, seq, dim)
    out = add_residual(branch_output)
    assert out.shape == (batch * streams, seq, dim)


def test_mhc_gradients_flow():
    from hyper_connections.hyper_connections import HyperConnections

    hc = HyperConnections(num_residual_streams=4, dim=64, mhc=True)
    x = torch.randn(8, 8, 64, requires_grad=True)

    branch_input, add_residual = hc(x)
    out = add_residual(branch_input)
    out.sum().backward()

    assert hc.H_res_logits.grad is not None
    assert hc.H_pre_logits.grad is not None
    assert hc.H_post_logits.grad is not None
    assert not torch.isnan(hc.H_res_logits.grad).any()
    assert not torch.isnan(hc.H_pre_logits.grad).any()
    assert not torch.isnan(hc.H_post_logits.grad).any()


def test_mhc_channel_first_H_res_constraints():
    from hyper_connections.hyper_connections_channel_first import HyperConnections
    from hyper_connections.hyper_connections import sinkhorn_log

    hc = HyperConnections(num_residual_streams=4, dim=64, mhc=True)
    H_res = sinkhorn_log(hc.H_res_logits, hc.sinkhorn_iters, hc.sinkhorn_tau)

    assert H_res.min().item() >= 0
    assert torch.allclose(
        H_res.sum(dim=-1),
        torch.ones(4, device=H_res.device, dtype=H_res.dtype),
        atol=1e-3,
    )
    assert torch.allclose(
        H_res.sum(dim=-2),
        torch.ones(4, device=H_res.device, dtype=H_res.dtype),
        atol=1e-3,
    )


def test_mhc_channel_first_H_pre_H_post_constraints():
    from hyper_connections.hyper_connections_channel_first import HyperConnections

    hc = HyperConnections(num_residual_streams=4, dim=64, mhc=True)
    H_pre = torch.softmax(hc.H_pre_logits, dim=-1)
    H_post = torch.softmax(hc.H_post_logits, dim=-1)

    assert H_pre.min().item() >= 0
    assert H_post.min().item() >= 0
    assert torch.allclose(
        H_pre.sum(),
        torch.ones((), device=H_pre.device, dtype=H_pre.dtype),
        atol=1e-6,
    )
    assert torch.allclose(
        H_post.sum(),
        torch.ones((), device=H_post.device, dtype=H_post.dtype),
        atol=1e-6,
    )


def test_mhc_channel_first_forward_shapes():
    from hyper_connections.hyper_connections_channel_first import HyperConnections

    streams, dim, batch, height, width = 4, 64, 2, 8, 8
    hc = HyperConnections(num_residual_streams=streams, dim=dim, mhc=True)
    x = torch.randn(batch * streams, dim, height, width)

    branch_input, add_residual = hc(x)
    assert branch_input.shape == (batch, dim, height, width)

    branch_output = torch.randn(batch, dim, height, width)
    out = add_residual(branch_output)
    assert out.shape == (batch * streams, dim, height, width)
