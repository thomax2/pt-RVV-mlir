"""GM-SplitCov v33: SD-GM frontend with Split-KalmanNet style backends."""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from .ekf_layer_knet_v33 import make_spd, wrap_to_pi


Tensor = torch.Tensor
HiddenDict = Dict[str, Tensor]


def scaled_logit_from_c(c: Tensor, c_min: float, c_max: float) -> Tensor:
    x = (c - float(c_min)) / max(float(c_max) - float(c_min), 1e-8)
    return torch.logit(x.clamp(1e-6, 1.0 - 1e-6))


def c_from_scaled_logit(xi: Tensor, c_min: float, c_max: float) -> Tensor:
    return float(c_min) + (float(c_max) - float(c_min)) * torch.sigmoid(xi)


def _masked_mean(x: Tensor, mask: Optional[Tensor], dim: int) -> Tensor:
    if mask is None:
        return x.mean(dim=dim)
    w = mask.to(device=x.device, dtype=x.dtype)
    while w.dim() < x.dim():
        w = w.unsqueeze(-1)
    return (x * w).sum(dim=dim) / w.sum(dim=dim).clamp_min(1.0)


def _masked_var(x: Tensor, mask: Optional[Tensor], dim: int) -> Tensor:
    mu = _masked_mean(x, mask, dim=dim)
    return _masked_mean((x - mu.unsqueeze(dim)) ** 2, mask, dim=dim)


def _masked_max(x: Tensor, mask: Optional[Tensor], dim: int) -> Tensor:
    if mask is None:
        return x.max(dim=dim).values
    w = mask.to(device=x.device).bool()
    while w.dim() < x.dim():
        w = w.unsqueeze(-1)
    return torch.where(w, x, torch.full_like(x, -1e9)).max(dim=dim).values


def _masked_min(x: Tensor, mask: Optional[Tensor], dim: int) -> Tensor:
    if mask is None:
        return x.min(dim=dim).values
    w = mask.to(device=x.device).bool()
    while w.dim() < x.dim():
        w = w.unsqueeze(-1)
    return torch.where(w, x, torch.full_like(x, 1e9)).min(dim=dim).values


class FrontendPointEncoder(nn.Module):
    """v32-compatible frontend encoder; backend encoders below are simpler."""

    def __init__(self, hidden_dim: int = 128, use_raw_stats: bool = True):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.use_raw_stats = bool(use_raw_stats)
        self.point_mlp = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.out_dim = 3 * hidden_dim + (10 if self.use_raw_stats else 0)

    def forward(self, z_norm: Tensor, mask: Optional[Tensor]) -> Tensor:
        z = z_norm if z_norm.shape[-1] == 2 else z_norm.transpose(-1, -2)
        x, y = z[..., 0:1], z[..., 1:2]
        valid = None if mask is None else mask.unsqueeze(-1).bool()
        r2 = x * x + y * y
        r = torch.sqrt(r2 + 1e-8)
        theta = torch.atan2(torch.where(r2 > 1e-10, y, torch.zeros_like(y)), torch.where(r2 > 1e-10, x, torch.ones_like(x)))
        feat_in = torch.cat([x, y, x.abs(), y.abs(), r, theta, x * x, y * y, x * y], dim=-1)
        if valid is not None:
            feat_in = torch.where(valid, feat_in, torch.zeros_like(feat_in))
        feat = self.point_mlp(feat_in)
        out = [
            _masked_max(feat, mask, dim=-2),
            _masked_mean(feat, mask, dim=-2),
            torch.sqrt(_masked_var(feat, mask, dim=-2).clamp_min(1e-8)),
        ]
        if self.use_raw_stats:
            z_mean = _masked_mean(z, mask, dim=-2)
            z_std = torch.sqrt(_masked_var(z, mask, dim=-2).clamp_min(1e-8))
            z_max = _masked_max(z, mask, dim=-2)
            z_min = _masked_min(z, mask, dim=-2)
            z_absmax = torch.maximum(z_max.abs(), z_min.abs())
            out.append(torch.cat([z_mean, z_std, z_max, z_min, z_absmax], dim=-1))
        return torch.cat(out, dim=-1)


class SimplePointEncoder(nn.Module):
    """Kept for compatibility/debug; current v33 backends use frontend point_feat + projection."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.out_dim = 2 * hidden_dim

    def forward(self, z_norm: Tensor, mask: Optional[Tensor]) -> Tensor:
        z = z_norm if z_norm.shape[-1] == 2 else z_norm.transpose(-1, -2)
        r = torch.sqrt((z ** 2).sum(dim=-1, keepdim=True).clamp_min(1e-8))
        feat_in = torch.cat([z, r], dim=-1)
        if mask is not None:
            feat_in = feat_in * mask.to(device=z.device, dtype=z.dtype).unsqueeze(-1)
        feat = self.point_mlp(feat_in)
        return torch.cat([_masked_mean(feat, mask, dim=1), _masked_max(feat, mask, dim=1)], dim=-1)


@dataclass
class GMKNetV33Config:
    s: float = 1.0 / 3.0
    c_min: float = 0.003
    c_max: float = 0.70
    tau_max: float = 100.0
    tau_min: float = 1.0
    alpha_base: float = 1.0
    logvar_min: float = -6.0
    logvar_max: float = 1.5
    state_dim: int = 12
    hidden_dim: int = 128
    backend_hidden_dim: int = 128
    kin_hidden_dim: int = 256
    front_point_dim: int = 394
    kin_point_proj_dim: int = 192
    shape_point_proj_dim: int = 192
    kin_obs_dim: int = 2
    n_ref: float = 40.0
    use_raw_stats: bool = True
    use_pi_gru: bool = False
    use_c_gru: bool = True
    use_kin_gru: bool = True
    use_shape_gru: bool = True
    sigma_jitter: float = 1e-5
    ch_jitter: float = 1e-6


class KinKNetBackend(nn.Module):
    def __init__(self, front_point_dim: int = 394, point_proj_dim: int = 192, hidden_dim: int = 256, obs_dim: int = 2):
        super().__init__()
        self.front_point_dim = int(front_point_dim)
        self.point_proj_dim = int(point_proj_dim)
        self.hidden_dim = int(hidden_dim)
        self.obs_dim = int(obs_dim)
        if self.obs_dim < 1 or self.obs_dim % 2 != 0:
            raise ValueError(f"obs_dim must be a positive even number, got {obs_dim}")
        self.point_proj = nn.Sequential(
            nn.Linear(self.front_point_dim, self.point_proj_dim),
            nn.LayerNorm(self.point_proj_dim),
            nn.SiLU(),
        )
        state_dim = 6
        innov_dim = self.obs_dim
        innov_delta_dim = self.obs_dim
        v_obs_dim = 4
        prev_update_dim = 5
        log_cr_dim = 5
        log_cy_dim = self.obs_dim
        k_dim = 5 * self.obs_dim
        gm_summary_dim = 5
        in_dim = (
            self.point_proj_dim
            + state_dim
            + innov_dim
            + innov_delta_dim
            + v_obs_dim
            + prev_update_dim
            + log_cr_dim
            + log_cy_dim
            + k_dim
            + gm_summary_dim
        )
        self.feature = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.deltaK_head = nn.Linear(self.hidden_dim, 5 * self.obs_dim)
        self.res_head = nn.Linear(self.hidden_dim, 5)
        scale_base = torch.tensor([[0.30, 0.30], [0.30, 0.30], [0.80, 0.80], [0.80, 0.80], [0.08, 0.08]], dtype=torch.float32)
        repeats = math.ceil(self.obs_dim / 2)
        self.register_buffer("scale", scale_base.repeat(1, repeats)[:, :self.obs_dim])
        self.register_buffer("res_scale", torch.tensor([0.04, 0.04, 0.15, 0.15, 0.10], dtype=torch.float32))
        nn.init.zeros_(self.deltaK_head.weight)
        nn.init.zeros_(self.deltaK_head.bias)
        nn.init.zeros_(self.res_head.weight)
        nn.init.zeros_(self.res_head.bias)

    def _frontend_point_feature(self, features: Dict[str, Tensor]) -> Tensor:
        front_feat = features.get("front_point_feat", None)
        if front_feat is None:
            front_feat = features.get("point_feat", None)
        if front_feat is None:
            raise KeyError("KinKNetBackend requires front_point_feat / point_feat from FrontendPointEncoder")
        if int(front_feat.shape[-1]) != self.front_point_dim:
            raise ValueError(f"KinKNetBackend expected front_point_dim={self.front_point_dim}, got {int(front_feat.shape[-1])}")
        return front_feat

    def forward(self, z_norm: Tensor, mask: Tensor, features: Dict[str, Tensor], h: Optional[Tensor] = None) -> Tuple[Dict[str, Tensor], Tensor]:
        front_feat = self._frontend_point_feature(features)
        B = front_feat.shape[0]
        if h is None:
            h = torch.zeros(B, self.gru.hidden_size, device=front_feat.device, dtype=front_feat.dtype)
        point = self.point_proj(front_feat)
        x = torch.cat([
            point,
            features["state_feat_kin"],
            features["innov"],
            features["innov_delta"],
            features["v_obs_feat"],
            features["prev_update_r"],
            features["log_diag_Cr_pred"],
            features["log_diag_Cy_mb"],
            features["K_mb"].reshape(B, -1),
            features["gm_summary"],
        ], dim=-1)
        x = torch.nan_to_num(x, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        h_new = self.gru(self.feature(x), h)
        raw = self.deltaK_head(h_new).view(B, 5, self.obs_dim)
        deltaK = self.scale.to(device=front_feat.device, dtype=front_feat.dtype).unsqueeze(0) * torch.tanh(raw)
        sat = (raw.abs() > 2.0).to(dtype=front_feat.dtype).mean()
        raw_res = self.res_head(h_new)
        delta_r_post = torch.tanh(raw_res) * self.res_scale.to(device=front_feat.device, dtype=front_feat.dtype).view(1, 5)
        res_sat = (torch.tanh(raw_res).abs() > 0.95).to(dtype=front_feat.dtype).mean()
        return {
            "deltaK": deltaK,
            "deltaK_abs_mean": deltaK.abs().mean(),
            "deltaK_saturation_frac": sat,
            "delta_r_post": delta_r_post,
            "raw_res": raw_res,
            "kin_res_saturation_frac": res_sat,
            "point_proj_norm_mean": point.norm(dim=1).mean(),
        }, h_new


class ShapePostNet(nn.Module):
    def __init__(self, front_point_dim: int = 394, point_proj_dim: int = 192, hidden_dim: int = 128):
        super().__init__()
        self.front_point_dim = int(front_point_dim)
        self.point_proj_dim = int(point_proj_dim)
        self.point_proj = nn.Sequential(
            nn.Linear(self.front_point_dim, self.point_proj_dim),
            nn.LayerNorm(self.point_proj_dim),
            nn.SiLU(),
        )
        in_dim = self.point_proj_dim + 3 + 3 + 2 + 5 + 3 + 5 + 1
        self.feature = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 3)
        self.register_buffer("scale", torch.tensor([0.06, 0.12, 0.08], dtype=torch.float32))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _frontend_point_feature(self, features: Dict[str, Tensor]) -> Tensor:
        front_feat = features.get("front_point_feat", None)
        if front_feat is None:
            front_feat = features.get("point_feat", None)
        if front_feat is None:
            raise KeyError("ShapePostNet requires front_point_feat / point_feat from FrontendPointEncoder")
        if int(front_feat.shape[-1]) != self.front_point_dim:
            raise ValueError(f"ShapePostNet expected front_point_dim={self.front_point_dim}, got {int(front_feat.shape[-1])}")
        return front_feat

    def forward(self, z_norm: Tensor, mask: Tensor, features: Dict[str, Tensor], h: Optional[Tensor] = None) -> Tuple[Dict[str, Tensor], Tensor]:
        front_feat = self._frontend_point_feature(features)
        B = front_feat.shape[0]
        if h is None:
            h = torch.zeros(B, self.gru.hidden_size, device=front_feat.device, dtype=front_feat.dtype)
        point = self.point_proj(front_feat)
        x = torch.cat([
            point,
            features["p_pred_feat"],
            features["p_base_feat"],
            features["innov"],
            features["kin_update_r"],
            features["log_diag_Cp_pred"],
            features["log_diag_Cr_plus"],
            features["log_n"],
        ], dim=-1)
        x = torch.nan_to_num(x, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        h_new = self.gru(self.feature(x), h)
        raw = self.head(h_new)
        delta = self.scale.to(device=front_feat.device, dtype=front_feat.dtype).view(1, 3) * torch.tanh(raw)
        p_base = features["p_base"]
        p_plus = torch.cat([
            wrap_to_pi(p_base[:, 0:1] + delta[:, 0:1]),
            (p_base[:, 1:2] + delta[:, 1:2]).clamp(0.5, 8.0),
            (p_base[:, 2:3] + delta[:, 2:3]).clamp(0.4, 4.0),
        ], dim=1)
        sat = (raw.abs() > 2.0).to(dtype=front_feat.dtype).mean()
        return {
            "delta_p": delta,
            "p_plus": p_plus,
            "shape_delta_yaw_abs_mean": delta[:, 0].abs().mean(),
            "shape_delta_l_abs_mean": delta[:, 1].abs().mean(),
            "shape_delta_w_abs_mean": delta[:, 2].abs().mean(),
            "shape_delta_saturation_frac": sat,
        }, h_new


class GMKNetV33(nn.Module):
    def __init__(self, cfg: Optional[GMKNetV33Config] = None):
        super().__init__()
        self.cfg = cfg or GMKNetV33Config()
        H = int(self.cfg.hidden_dim)
        self.point_encoder = FrontendPointEncoder(H, use_raw_stats=self.cfg.use_raw_stats)
        pi_in = self.point_encoder.out_dim + 2 + 1
        c_in = self.point_encoder.out_dim + 1 + int(self.cfg.state_dim)
        self.pi_encoder = nn.Sequential(nn.Linear(pi_in, H), nn.LayerNorm(H), nn.SiLU(), nn.Linear(H, H), nn.LayerNorm(H), nn.SiLU())
        self.c_encoder = nn.Sequential(nn.Linear(c_in, H), nn.LayerNorm(H), nn.SiLU(), nn.Linear(H, H), nn.LayerNorm(H), nn.SiLU())
        self.pi_gru = nn.GRUCell(H, H) if self.cfg.use_pi_gru else None
        self.c_gru = nn.GRUCell(H, H) if self.cfg.use_c_gru else None
        self.pi_head = nn.Sequential(nn.Linear(H, H), nn.LayerNorm(H), nn.SiLU(), nn.Linear(H, 4))
        self.tau_head = nn.Sequential(nn.Linear(H, H // 2), nn.SiLU(), nn.Linear(H // 2, 1))
        self.c_xi_head = nn.Sequential(nn.Linear(H, H), nn.LayerNorm(H), nn.SiLU(), nn.Linear(H, 2))
        self.c_logvar_head = nn.Sequential(nn.Linear(H, H // 2), nn.SiLU(), nn.Linear(H // 2, 2))
        self.kin_backend = KinKNetBackend(
            front_point_dim=int(self.cfg.front_point_dim),
            point_proj_dim=int(self.cfg.kin_point_proj_dim),
            hidden_dim=int(self.cfg.kin_hidden_dim),
            obs_dim=int(self.cfg.kin_obs_dim),
        )
        self.shape_post = ShapePostNet(
            front_point_dim=int(self.cfg.front_point_dim),
            point_proj_dim=int(self.cfg.shape_point_proj_dim),
            hidden_dim=int(self.cfg.backend_hidden_dim),
        )
        last = self.c_logvar_head[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    @property
    def ilr_basis(self) -> Tensor:
        return torch.tensor([
            [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(6.0), 1.0 / math.sqrt(12.0)],
            [-1.0 / math.sqrt(2.0), 1.0 / math.sqrt(6.0), 1.0 / math.sqrt(12.0)],
            [0.0, -2.0 / math.sqrt(6.0), 1.0 / math.sqrt(12.0)],
            [0.0, 0.0, -3.0 / math.sqrt(12.0)],
        ], dtype=torch.float32)

    def init_hidden(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> HiddenDict:
        h: HiddenDict = {
            "kin": torch.zeros(batch_size, self.kin_backend.gru.hidden_size, device=device, dtype=dtype),
            "shape": torch.zeros(batch_size, self.shape_post.gru.hidden_size, device=device, dtype=dtype),
        }
        if self.pi_gru is not None:
            h["pi"] = torch.zeros(batch_size, int(self.cfg.hidden_dim), device=device, dtype=dtype)
        if self.c_gru is not None:
            h["c"] = torch.zeros(batch_size, int(self.cfg.hidden_dim), device=device, dtype=dtype)
        return h

    @staticmethod
    def detach_hidden(hidden: Optional[HiddenDict]) -> Optional[HiddenDict]:
        if hidden is None:
            return None
        return {k: v.detach() for k, v in hidden.items()}

    @staticmethod
    def default_prior(batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> Tuple[Tensor, Tensor]:
        return (
            torch.full((batch_size, 2), 0.08, device=device, dtype=dtype),
            torch.full((batch_size, 4), 0.25, device=device, dtype=dtype),
        )

    def _q_count(self, mask: Optional[Tensor], batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if mask is None:
            n = torch.full((batch_size,), float(self.cfg.n_ref), device=device, dtype=dtype)
        else:
            n = mask.sum(dim=1).to(dtype=dtype).clamp_min(1.0)
        return (torch.log1p(n) / math.log1p(float(self.cfg.n_ref))).clamp(0.0, 1.0).unsqueeze(1)

    def make_wen_gmm_from_c(self, c_mean: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        cfg = self.cfg
        c_l = c_mean[:, 0].clamp(cfg.c_min, cfg.c_max)
        c_w = c_mean[:, 1].clamp(cfg.c_min, cfg.c_max)
        d_l = 1.0 - torch.sqrt(c_l)
        d_w = 1.0 - torch.sqrt(c_w)
        zero = torch.zeros_like(d_l)
        h_hat_set = torch.stack([
            torch.stack([d_l, zero], dim=1),
            torch.stack([zero, d_w], dim=1),
            torch.stack([-d_l, zero], dim=1),
            torch.stack([zero, -d_w], dim=1),
        ], dim=2)
        B = c_mean.shape[0]
        s = float(cfg.s)
        Ch_set = torch.zeros(B, 2, 2, 4, device=c_mean.device, dtype=c_mean.dtype)
        Ch_set[:, 0, 0, 0] = s * c_l
        Ch_set[:, 1, 1, 0] = s
        Ch_set[:, 0, 0, 2] = s * c_l
        Ch_set[:, 1, 1, 2] = s
        Ch_set[:, 0, 0, 1] = s
        Ch_set[:, 1, 1, 1] = s * c_w
        Ch_set[:, 0, 0, 3] = s
        Ch_set[:, 1, 1, 3] = s * c_w
        gmm_params = torch.stack([d_l, d_w, torch.full_like(d_l, s), s * c_w, torch.full_like(d_l, s), s * c_l], dim=1)
        return h_hat_set, Ch_set, gmm_params

    def centroid_moments(self, alpha_pi: Tensor, c_xi: Tensor, c_logvar: Tensor, detach_cov: bool = True) -> Tuple[Tensor, Tensor]:
        cfg = self.cfg
        B = alpha_pi.shape[0]
        device, dtype = alpha_pi.device, alpha_pi.dtype
        basis = self.ilr_basis.to(device=device, dtype=dtype)
        mu_u = torch.matmul(torch.digamma(alpha_pi), basis)
        trig = torch.polygamma(1, alpha_pi).clamp_min(1e-7)
        Sigma_u = torch.einsum("ji,bj,jk->bik", basis, trig, basis)
        var_xi = torch.exp(c_logvar).clamp_min(1e-8)
        Sigma = torch.zeros(B, 5, 5, device=device, dtype=dtype)
        Sigma[:, :3, :3] = Sigma_u
        Sigma[:, 3, 3] = var_xi[:, 0]
        Sigma[:, 4, 4] = var_xi[:, 1]
        L = torch.linalg.cholesky(make_spd(Sigma, eps=float(cfg.sigma_jitter), max_abs=1e4))
        n = 5
        mu = torch.cat([mu_u, c_xi], dim=1)
        offsets = math.sqrt(float(n)) * L.transpose(1, 2)
        pts = torch.cat([mu.unsqueeze(1) + offsets, mu.unsqueeze(1) - offsets], dim=1)
        logits = torch.einsum("bmc,jc->bmj", pts[:, :, :3], basis)
        pi = torch.softmax(logits, dim=-1)
        c = c_from_scaled_logit(pts[:, :, 3:5], cfg.c_min, cfg.c_max).clamp(cfg.c_min, cfg.c_max)
        d_l = 1.0 - torch.sqrt(c[:, :, 0].clamp_min(cfg.c_min))
        d_w = 1.0 - torch.sqrt(c[:, :, 1].clamp_min(cfg.c_min))
        h = torch.stack([(pi[:, :, 0] - pi[:, :, 2]) * d_l, (pi[:, :, 1] - pi[:, :, 3]) * d_w], dim=-1)
        m_h = h.mean(dim=1)
        dh = h - m_h.unsqueeze(1)
        C_h = torch.einsum("bmi,bmj->bij", dh, dh) / float(2 * n)
        C_h = make_spd(C_h, eps=float(cfg.ch_jitter), max_abs=1e4)
        return m_h, C_h.detach() if detach_cov else C_h

    def measurement_step(
        self,
        z_norm: Tensor,
        sensor_view: Tensor,
        state_feat: Tensor,
        mask: Optional[Tensor],
        hidden: Optional[Union[HiddenDict, Tensor]] = None,
        detach_ch_cov: bool = True,
    ) -> Tuple[Dict[str, Tensor], HiddenDict]:
        B = z_norm.shape[0]
        hidden_dict: HiddenDict = {} if hidden is None else ({"c": hidden} if isinstance(hidden, torch.Tensor) else hidden)
        point_feat = self.point_encoder(z_norm, mask)
        qN = self._q_count(mask, B, z_norm.device, z_norm.dtype)
        pi_feat = self.pi_encoder(torch.cat([point_feat, sensor_view, qN], dim=1))
        h_pi = self.pi_gru(pi_feat, hidden_dict.get("pi", torch.zeros_like(pi_feat))) if self.pi_gru is not None else pi_feat
        c_feat = self.c_encoder(torch.cat([point_feat, qN, state_feat], dim=1))
        h_c = self.c_gru(c_feat, hidden_dict.get("c", torch.zeros_like(c_feat))) if self.c_gru is not None else c_feat
        pi_logits = self.pi_head(h_pi)
        pi_mu = torch.softmax(pi_logits, dim=1)
        tau_raw = torch.sigmoid(self.tau_head(h_pi))
        tau = float(self.cfg.tau_min) + qN * float(self.cfg.tau_max) * tau_raw
        alpha_pi = float(self.cfg.alpha_base) + tau * pi_mu
        alpha0 = alpha_pi.sum(dim=1, keepdim=True)
        s_pi = (torch.log1p(alpha0) / math.log1p(4.0 * float(self.cfg.alpha_base) + float(self.cfg.tau_min) + float(self.cfg.tau_max))).clamp(0.0, 1.0)
        c_xi = self.c_xi_head(h_c)
        c_mean = c_from_scaled_logit(c_xi, self.cfg.c_min, self.cfg.c_max)
        raw_lv = self.c_logvar_head(h_c)
        c_logvar = self.cfg.logvar_min + (self.cfg.logvar_max - self.cfg.logvar_min) * torch.sigmoid(raw_lv)
        c_var = torch.exp(c_logvar)
        q_c = 1.0 / (1.0 + c_var)
        h_hat_set, Ch_set, gmm_params = self.make_wen_gmm_from_c(c_mean)
        m_h, C_h = self.centroid_moments(alpha_pi, c_xi, c_logvar, detach_cov=detach_ch_cov)
        new_hidden: HiddenDict = {k: v for k, v in hidden_dict.items() if k in {"kin", "shape"}}
        if self.pi_gru is not None:
            new_hidden["pi"] = h_pi
        if self.c_gru is not None:
            new_hidden["c"] = h_c
        return {
            "point_feat": point_feat,
            "front_point_feat": point_feat,
            "pi_logits": pi_logits,
            "pi_mu": pi_mu,
            "pi": pi_mu,
            "tau": tau.squeeze(1),
            "alpha_pi": alpha_pi,
            "alpha0_pi": alpha0.squeeze(1),
            "s_pi": s_pi.squeeze(1),
            "qN": qN.squeeze(1),
            "c_xi": c_xi,
            "c_mean": c_mean,
            "c_logvar": c_logvar,
            "c_var": c_var,
            "q_c": q_c,
            "h_hat_set": h_hat_set,
            "Ch_set": Ch_set,
            "gmm_params": gmm_params,
            "m_h": m_h,
            "C_h": C_h,
            "pi_hidden": h_pi,
            "c_hidden": h_c,
        }, new_hidden

    def forward(self, z_norm: Tensor, sensor_view: Tensor, state_feat: Tensor, mask: Optional[Tensor], hidden: Optional[HiddenDict] = None, detach_ch_cov: bool = True) -> Tuple[Dict[str, Tensor], HiddenDict]:
        return self.measurement_step(z_norm, sensor_view, state_feat, mask, hidden, detach_ch_cov)


def build_default_model_v33(**kwargs) -> GMKNetV33:
    valid = {f.name for f in fields(GMKNetV33Config)}
    return GMKNetV33(GMKNetV33Config(**{k: v for k, v in kwargs.items() if k in valid}))
