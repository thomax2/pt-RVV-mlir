"""Functional EOT custom operators used as stable torch.export boundaries."""
from __future__ import annotations

import math

import torch
from torch import Tensor

SCHEMA_VERSION = 1

if not hasattr(torch.library, "custom_op"):
    raise RuntimeError(
        "GM-EOT export requires a PyTorch build with torch.library.custom_op "
        "and torch.export (the installed interpreter is too old)")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@torch.library.custom_op("eot::point_features", mutates_args=())
def point_features(points: Tensor, mask: Tensor, radius_epsilon: float,
                   theta_epsilon: float) -> Tensor:
    _require(points.ndim == 3 and points.shape[-1] == 2,
             f"points must have shape [B,N,2], got {tuple(points.shape)}")
    _require(mask.ndim == 2 and tuple(mask.shape) == tuple(points.shape[:2]),
             "mask must have shape [B,N]")
    _require(points.dtype == torch.float32 and mask.dtype == torch.bool,
             "point_features requires f32 points and bool mask")
    x, y = points[..., 0:1], points[..., 1:2]
    r2 = x * x + y * y
    radius = torch.sqrt(r2 + radius_epsilon)
    valid_angle = r2 > theta_epsilon
    theta = torch.atan2(torch.where(valid_angle, y, torch.zeros_like(y)),
                        torch.where(valid_angle, x, torch.ones_like(x)))
    result = torch.cat((x, y, x.abs(), y.abs(), radius, theta,
                        x * x, y * y, x * y), dim=-1)
    return torch.where(mask.unsqueeze(-1), result, torch.zeros_like(result))


@point_features.register_fake
def _point_features_fake(points: Tensor, mask: Tensor, radius_epsilon: float,
                         theta_epsilon: float) -> Tensor:
    return points.new_empty((*points.shape[:-1], 9))


def _check_masked(input: Tensor, mask: Tensor) -> None:
    _require(input.ndim == 3, "input must have shape [B,N,C]")
    _require(mask.ndim == 2 and tuple(mask.shape) == tuple(input.shape[:2]),
             "mask must have shape [B,N]")
    _require(input.dtype == torch.float32 and mask.dtype == torch.bool,
             "masked operators require f32 input and bool mask")


@torch.library.custom_op("eot::masked_mean", mutates_args=())
def masked_mean(input: Tensor, mask: Tensor) -> Tensor:
    _check_masked(input, mask)
    weight = mask.to(input.dtype).unsqueeze(-1)
    return (input * weight).sum(1) / weight.sum(1).clamp_min(1.0)


@masked_mean.register_fake
def _masked_mean_fake(input: Tensor, mask: Tensor) -> Tensor:
    return input.new_empty((input.shape[0], input.shape[2]))


@torch.library.custom_op("eot::masked_variance", mutates_args=())
def masked_variance(input: Tensor, mask: Tensor) -> Tensor:
    _check_masked(input, mask)
    weight = mask.to(input.dtype).unsqueeze(-1)
    count = weight.sum(1).clamp_min(1.0)
    mean = (input * weight).sum(1) / count
    delta = input - mean.unsqueeze(1)
    return (delta.square() * weight).sum(1) / count


@masked_variance.register_fake
def _masked_variance_fake(input: Tensor, mask: Tensor) -> Tensor:
    return input.new_empty((input.shape[0], input.shape[2]))


@torch.library.custom_op("eot::masked_max", mutates_args=())
def masked_max(input: Tensor, mask: Tensor, empty_value: float) -> Tensor:
    _check_masked(input, mask)
    fill = torch.full_like(input, empty_value)
    return torch.where(mask.unsqueeze(-1), input, fill).amax(1)


@masked_max.register_fake
def _masked_max_fake(input: Tensor, mask: Tensor, empty_value: float) -> Tensor:
    return input.new_empty((input.shape[0], input.shape[2]))


@torch.library.custom_op("eot::masked_min", mutates_args=())
def masked_min(input: Tensor, mask: Tensor, empty_value: float) -> Tensor:
    _check_masked(input, mask)
    fill = torch.full_like(input, empty_value)
    return torch.where(mask.unsqueeze(-1), input, fill).amin(1)


@masked_min.register_fake
def _masked_min_fake(input: Tensor, mask: Tensor, empty_value: float) -> Tensor:
    return input.new_empty((input.shape[0], input.shape[2]))


@torch.library.custom_op("eot::gru_cell", mutates_args=())
def gru_cell(x: Tensor, h_prev: Tensor, weight_ih: Tensor, weight_hh: Tensor,
             bias_ih: Tensor, bias_hh: Tensor) -> Tensor:
    _require(x.ndim == 2 and h_prev.ndim == 2, "x and h_prev must be rank 2")
    _require(x.shape[0] == h_prev.shape[0], "x and h_prev batch sizes must match")
    hidden = h_prev.shape[1]
    _require(weight_ih.shape == (3 * hidden, x.shape[1]), "invalid weight_ih")
    _require(weight_hh.shape == (3 * hidden, hidden), "invalid weight_hh")
    _require(bias_ih.shape == (3 * hidden,) and bias_hh.shape == (3 * hidden,),
             "invalid GRU bias")
    gi = torch.nn.functional.linear(x, weight_ih, bias_ih)
    gh = torch.nn.functional.linear(h_prev, weight_hh, bias_hh)
    i_r, i_z, i_n = gi.chunk(3, 1)
    h_r, h_z, h_n = gh.chunk(3, 1)
    reset = torch.sigmoid(i_r + h_r)
    update = torch.sigmoid(i_z + h_z)
    candidate = torch.tanh(i_n + reset * h_n)
    return (1.0 - update) * candidate + update * h_prev


@gru_cell.register_fake
def _gru_cell_fake(x: Tensor, h_prev: Tensor, weight_ih: Tensor,
                   weight_hh: Tensor, bias_ih: Tensor, bias_hh: Tensor) -> Tensor:
    return h_prev.new_empty(h_prev.shape)


@torch.library.custom_op("eot::gm_parameterize", mutates_args=())
def gm_parameterize(pi_logits: Tensor, tau_logits: Tensor, c_xi: Tensor,
                    raw_logvar: Tensor, qN: Tensor, c_min: float, c_max: float,
                    tau_min: float, tau_max: float, alpha_base: float,
                    logvar_min: float, logvar_max: float
                    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor,
                               Tensor, Tensor, Tensor, Tensor]:
    _require(pi_logits.ndim == 2 and pi_logits.shape[1] == 4,
             "pi_logits must have shape [B,4]")
    _require(tau_logits.ndim == 2 and tau_logits.shape[1] == 1,
             "tau_logits must have shape [B,1]")
    _require(c_xi.ndim == 2 and c_xi.shape[1] == 2,
             "c_xi must have shape [B,2]")
    _require(raw_logvar.shape == c_xi.shape, "raw_logvar must have shape [B,2]")
    _require(qN.ndim == 2 and qN.shape[1] == 1, "qN must have shape [B,1]")
    _require(all(t.shape[0] == pi_logits.shape[0]
                 for t in (tau_logits, c_xi, raw_logvar, qN)),
             "gm_parameterize batch sizes must match")
    pi = torch.softmax(pi_logits, dim=1)
    tau = tau_min + qN * tau_max * torch.sigmoid(tau_logits)
    alpha_pi = alpha_base + tau * pi
    alpha0 = alpha_pi.sum(1, keepdim=True)
    denominator = math.log1p(4.0 * alpha_base + tau_min + tau_max)
    s_pi = (torch.log1p(alpha0) / denominator).clamp(0.0, 1.0)
    c_mean = c_min + (c_max - c_min) * torch.sigmoid(c_xi)
    c_logvar = logvar_min + (logvar_max - logvar_min) * torch.sigmoid(raw_logvar)
    c_var = torch.exp(c_logvar)
    q_c = 1.0 / (1.0 + c_var)
    return pi, tau, alpha_pi, alpha0, s_pi, c_mean, c_logvar, c_var, q_c


@gm_parameterize.register_fake
def _gm_parameterize_fake(pi_logits: Tensor, tau_logits: Tensor, c_xi: Tensor,
                          raw_logvar: Tensor, qN: Tensor, c_min: float,
                          c_max: float, tau_min: float, tau_max: float,
                          alpha_base: float, logvar_min: float,
                          logvar_max: float
                          ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor,
                                     Tensor, Tensor, Tensor, Tensor]:
    batch = pi_logits.shape[0]
    return (pi_logits.new_empty((batch, 4)), pi_logits.new_empty((batch, 1)),
            pi_logits.new_empty((batch, 4)), pi_logits.new_empty((batch, 1)),
            pi_logits.new_empty((batch, 1)), pi_logits.new_empty((batch, 2)),
            pi_logits.new_empty((batch, 2)), pi_logits.new_empty((batch, 2)),
            pi_logits.new_empty((batch, 2)))


ALL_OPS = (point_features, masked_mean, masked_variance, masked_max,
           masked_min, gru_cell, gm_parameterize)
