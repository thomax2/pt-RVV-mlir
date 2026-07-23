"""Deployment-only wrapper for the GM v33 neural measurement frontend."""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn

from . import eot_export_ops as eot


def _tosa_layer_norm(value: Tensor, layer: nn.LayerNorm,
                     epsilon: Tensor) -> Tensor:
    """LayerNorm expressed with Torch ops covered by the TOSA backend."""
    if len(layer.normalized_shape) != 1:
        raise ValueError("EOT export supports only one-dimensional LayerNorm")
    mean = torch.mean(value, dim=-1, keepdim=True)
    centered = value - mean
    variance = torch.mean(centered * centered, dim=-1, keepdim=True)
    normalized = centered * torch.rsqrt(variance + epsilon)
    if layer.elementwise_affine:
        normalized = normalized * layer.weight
        if layer.bias is not None:
            normalized = normalized + layer.bias
    return normalized


def _tosa_silu(value: Tensor) -> Tensor:
    """SiLU decomposed to TOSA-covered elementwise operations."""
    return value * torch.sigmoid(value)


def _run_tosa_sequential(sequence: nn.Sequential, value: Tensor,
                         layer_norm_epsilon: Tensor) -> Tensor:
    for layer in sequence:
        if isinstance(layer, nn.LayerNorm):
            value = _tosa_layer_norm(value, layer, layer_norm_epsilon)
        elif isinstance(layer, nn.SiLU):
            value = _tosa_silu(value)
        else:
            value = layer(value)
    return value


class EotFrontendExportModule(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        cfg = model.cfg
        if cfg.use_pi_gru:
            raise ValueError("first EOT frontend ABI requires use_pi_gru=False")
        if not cfg.use_c_gru or model.c_gru is None:
            raise ValueError("first EOT frontend ABI requires use_c_gru=True")
        if (cfg.hidden_dim, cfg.state_dim, cfg.front_point_dim) != (128, 12, 394):
            raise ValueError("expected hidden_dim=128, state_dim=12, front_point_dim=394")

        # 提取关键组件的引用
        # self. 是 torch mlir 标准算子，可以自动转换为linalg、tosa
        self.cfg = cfg
        self.point_mlp = model.point_encoder.point_mlp
        self.pi_encoder = model.pi_encoder
        self.c_encoder = model.c_encoder
        self.c_gru = model.c_gru
        self.pi_head = model.pi_head
        self.tau_head = model.tau_head
        self.c_xi_head = model.c_xi_head
        self.c_logvar_head = model.c_logvar_head
        self.register_buffer(
            "_layer_norm_epsilon", torch.tensor(1.0e-5, dtype=torch.float32),
            persistent=False)
        self.register_buffer(
            "_variance_floor", torch.tensor(1.0e-8, dtype=torch.float32),
            persistent=False)
        self.register_buffer(
            "_zero", torch.tensor(0.0, dtype=torch.float32),
            persistent=False)
        self.register_buffer(
            "_one", torch.tensor(1.0, dtype=torch.float32),
            persistent=False)
        self.register_buffer(
            "_q_count_scale",
            torch.tensor(1.0 / math.log1p(float(cfg.n_ref)),
                         dtype=torch.float32),
            persistent=False)

    def forward(self, points: Tensor, sensor_view: Tensor, state_feat: Tensor,
                mask: Tensor, c_hidden_prev: Tensor):

        # eot 是自定义算子，在mlir中生成 torch.operator "eot.*" 需要自己写pass转换

        point9 = eot.point_features(points, mask, 1e-8, 1e-10)
        feat = _run_tosa_sequential(
            self.point_mlp, point9, self._layer_norm_epsilon)
        feat_max = eot.masked_max(feat, mask, -1e9)
        feat_mean = eot.masked_mean(feat, mask)
        feat_var = eot.masked_variance(feat, mask)
        feat_std = torch.sqrt(torch.maximum(
            feat_var, self._variance_floor))
        parts = [feat_max, feat_mean, feat_std]
        if self.cfg.use_raw_stats:
            z_mean = eot.masked_mean(points, mask)
            z_var = eot.masked_variance(points, mask)
            z_std = torch.sqrt(torch.maximum(
                z_var, self._variance_floor))
            z_max = eot.masked_max(points, mask, -1e9)
            z_min = eot.masked_min(points, mask, 1e9)
            z_absmax = torch.maximum(z_max.abs(), z_min.abs())
            parts.append(torch.cat((z_mean, z_std, z_max, z_min, z_absmax), 1))
        point_feat = torch.cat(parts, 1)
        torch._assert(point_feat.shape[1] == self.cfg.front_point_dim,
                      "unexpected point feature width")
        count = torch.maximum(
            torch.sum(mask, dim=1, dtype=torch.float32), self._one)
        qN = torch.log1p(count) * self._q_count_scale
        qN = torch.minimum(
            torch.maximum(qN, self._zero), self._one).unsqueeze(1)
        pi_feat = _run_tosa_sequential(
            self.pi_encoder, torch.cat((point_feat, sensor_view, qN), 1),
            self._layer_norm_epsilon)
        c_feat = _run_tosa_sequential(
            self.c_encoder, torch.cat((point_feat, qN, state_feat), 1),
            self._layer_norm_epsilon)
        c_hidden_next = eot.gru_cell(
            c_feat, c_hidden_prev, self.c_gru.weight_ih, self.c_gru.weight_hh,
            self.c_gru.bias_ih, self.c_gru.bias_hh)
        pi_logits = _run_tosa_sequential(
            self.pi_head, pi_feat, self._layer_norm_epsilon)
        tau_logits = _run_tosa_sequential(
            self.tau_head, pi_feat, self._layer_norm_epsilon)
        c_xi = _run_tosa_sequential(
            self.c_xi_head, c_hidden_next, self._layer_norm_epsilon)
        raw_logvar = _run_tosa_sequential(
            self.c_logvar_head, c_hidden_next, self._layer_norm_epsilon)
        outputs = eot.gm_parameterize(
            pi_logits, tau_logits, c_xi, raw_logvar, qN,
            float(self.cfg.c_min), float(self.cfg.c_max),
            float(self.cfg.tau_min), float(self.cfg.tau_max),
            float(self.cfg.alpha_base), float(self.cfg.logvar_min),
            float(self.cfg.logvar_max))
        pi, _, _, _, _, c_mean, c_logvar, _, _ = outputs
        return pi, c_mean, c_logvar, c_hidden_next
