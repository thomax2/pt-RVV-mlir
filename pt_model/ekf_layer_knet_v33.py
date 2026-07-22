"""KalmanNet-style MEM backend utilities for GM-SplitCov v33.

v33 keeps the analytic GM-MEM measurement construction, then lets a small
network learn only a bounded Kalman-gain correction.  There are no learned
covariance adapters, pseudo-measurements, covariance floors, or prior dynamics
corrections in this module.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


Tensor = torch.Tensor


def wrap_to_pi(angle: Tensor) -> Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _eye(batch: int, dim: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.eye(dim, device=device, dtype=dtype).unsqueeze(0).expand(batch, dim, dim)


def symmetrize(A: Tensor) -> Tensor:
    return 0.5 * (A + A.transpose(-1, -2))


def make_spd(A: Tensor, eps: float = 1e-6, max_abs: float = 1e6) -> Tensor:
    A = torch.nan_to_num(symmetrize(A), nan=0.0, posinf=max_abs, neginf=-max_abs).clamp(-max_abs, max_abs)
    B, n, _ = A.shape
    return symmetrize(A) + float(eps) * _eye(B, n, A.device, A.dtype)


def safe_cholesky(A: Tensor, eps: float = 1e-5, max_abs: float = 1e6) -> Tensor:
    A = make_spd(A, eps=eps, max_abs=max_abs)
    B, n, _ = A.shape
    eye = _eye(B, n, A.device, A.dtype)
    jitter = float(eps)
    for _ in range(7):
        L, info = torch.linalg.cholesky_ex(A + jitter * eye)
        if bool((info == 0).all()):
            return L
        jitter *= 10.0
    vals, vecs = torch.linalg.eigh(A.detach())
    vals = torch.nan_to_num(vals, nan=jitter, posinf=max_abs, neginf=jitter).clamp(jitter, max_abs)
    A_proj = symmetrize(torch.bmm(torch.bmm(vecs, torch.diag_embed(vals)), vecs.transpose(1, 2)))
    return torch.linalg.cholesky(A_proj + jitter * eye)


def solve_spd(A: Tensor, Bmat: Tensor, eps: float = 1e-5, max_abs: float = 1e6) -> Tuple[Tensor, Tensor]:
    A = make_spd(A, eps=eps, max_abs=max_abs)
    L = safe_cholesky(A, eps=eps, max_abs=max_abs)
    X = torch.cholesky_solve(Bmat, L)
    logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1).clamp_min(1e-12)).sum(dim=-1)
    logdet = torch.nan_to_num(logdet, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
    return X, logdet


def masked_mean_points(Z: Tensor, mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
    B, _, N = Z.shape
    if mask is None:
        n_eff = torch.full((B,), float(max(N, 1)), device=Z.device, dtype=Z.dtype)
        return Z.mean(dim=2), n_eff
    m = mask.to(device=Z.device, dtype=Z.dtype)
    n_eff = m.sum(dim=1).clamp_min(1.0)
    mean = (Z * m.unsqueeze(1)).sum(dim=2) / n_eff.view(B, 1)
    return mean, n_eff


def mixture_moments(pi: Tensor, h_hat_set: Tensor, Ch_set: Tensor) -> Tuple[Tensor, Tensor]:
    pi = pi.clamp_min(1e-8)
    pi = pi / pi.sum(dim=1, keepdim=True).clamp_min(1e-8)
    h_mean = torch.sum(h_hat_set * pi.unsqueeze(1), dim=2)
    diff = h_hat_set - h_mean.unsqueeze(2)
    Ch_mix = torch.zeros(pi.shape[0], 2, 2, device=pi.device, dtype=pi.dtype)
    for j in range(4):
        d = diff[:, :, j:j + 1]
        Ch_mix = Ch_mix + pi[:, j].view(-1, 1, 1) * (Ch_set[:, :, :, j] + torch.bmm(d, d.transpose(1, 2)))
    return h_mean, make_spd(Ch_mix, eps=1e-7, max_abs=1e4)


def gm_auxiliary_variables(p: Tensor, Cp: Tensor, Ch: Tensor, h_mean: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    yaw = p[:, 0]
    length = p[:, 1].clamp_min(0.5)
    width = p[:, 2].clamp_min(0.4)
    c, s = torch.cos(yaw), torch.sin(yaw)
    rot = torch.stack([torch.stack([c, -s], dim=-1), torch.stack([s, c], dim=-1)], dim=-2)
    D = torch.diag_embed(torch.stack([length / 2.0, width / 2.0], dim=1))
    Sshape = torch.bmm(rot, D)
    S1, S2 = Sshape[:, 0:1, :], Sshape[:, 1:2, :]
    half = 0.5
    J1 = torch.stack([
        torch.stack([-half * length * s, half * c, torch.zeros_like(length)], dim=-1),
        torch.stack([-half * width * c, torch.zeros_like(length), -half * s], dim=-1),
    ], dim=-2)
    J2 = torch.stack([
        torch.stack([half * length * c, half * s, torch.zeros_like(length)], dim=-1),
        torch.stack([-half * width * s, torch.zeros_like(length), half * c], dim=-1),
    ], dim=-2)
    CI = torch.bmm(torch.bmm(Sshape, Ch), Sshape.transpose(1, 2))
    Ch_aug = Ch + torch.bmm(h_mean.unsqueeze(2), h_mean.unsqueeze(1))

    def trace_bmm(A: Tensor, Bm: Tensor, C: Tensor, Dm: Tensor) -> Tensor:
        M = torch.bmm(torch.bmm(torch.bmm(A, Bm), C), Dm)
        return torch.diagonal(M, dim1=-2, dim2=-1).sum(-1)

    CII_00 = trace_bmm(Cp, J1.transpose(1, 2), Ch_aug, J1)
    CII_01 = trace_bmm(Cp, J2.transpose(1, 2), Ch_aug, J1)
    CII_10 = trace_bmm(Cp, J1.transpose(1, 2), Ch_aug, J2)
    CII_11 = trace_bmm(Cp, J2.transpose(1, 2), Ch_aug, J2)
    CII = torch.stack([torch.stack([CII_00, CII_01], dim=-1), torch.stack([CII_10, CII_11], dim=-1)], dim=-2)
    return CI, CII, Sshape


def compute_kinematic_mb_global_mean(
    Z: Tensor,
    mask: Optional[Tensor],
    r_pred: Tensor,
    p_pred: Tensor,
    Cr_pred: Tensor,
    Cp_pred: Tensor,
    Cv: Tensor,
    pi_mu: Tensor,
    h_hat_set: Tensor,
    Ch_set: Tensor,
    r_mean_gamma: float = 0.7,
) -> Dict[str, Tensor]:
    """Compute the analytic GM-MEM mean observation and baseline Kalman update."""
    B = r_pred.shape[0]
    device, dtype = r_pred.device, r_pred.dtype
    H = torch.tensor([[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.]], device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
    y_mean, n_eff = masked_mean_points(Z, mask)
    h_mean, Ch_mix = mixture_moments(pi_mu, h_hat_set, Ch_set)
    CI, CII, Sshape = gm_auxiliary_variables(p_pred, Cp_pred, Ch_mix, h_mean)
    center_offset = torch.bmm(Sshape, h_mean.unsqueeze(-1)).squeeze(-1)
    center_meas = y_mean - center_offset
    ybar = torch.bmm(H, r_pred.unsqueeze(-1)).squeeze(-1) + center_offset
    innov = y_mean - ybar
    R_point = make_spd(CI + CII + Cv, eps=1e-7, max_abs=1e5)
    denom = n_eff.view(B, 1, 1).clamp_min(1.0).pow(float(r_mean_gamma))
    R_mean = make_spd(R_point / denom, eps=1e-7, max_abs=1e5)
    Cy_mb = make_spd(torch.bmm(torch.bmm(H, Cr_pred), H.transpose(1, 2)) + R_mean, eps=1e-5, max_abs=1e6)
    CrH = torch.bmm(Cr_pred, H.transpose(1, 2))
    K_mb = solve_spd(Cy_mb, CrH.transpose(1, 2), eps=1e-5, max_abs=1e6)[0].transpose(1, 2)
    r_mb = r_pred + torch.bmm(K_mb, innov.unsqueeze(-1)).squeeze(-1)
    I = _eye(B, 5, device, dtype)
    A = I - torch.bmm(K_mb, H)
    Cr_mb = make_spd(torch.bmm(torch.bmm(A, Cr_pred), A.transpose(1, 2)) + torch.bmm(torch.bmm(K_mb, R_mean), K_mb.transpose(1, 2)), eps=1e-6, max_abs=1e6)
    solve_inn, _ = solve_spd(Cy_mb, innov.unsqueeze(-1), eps=1e-5, max_abs=1e6)
    nis_mb = torch.bmm(innov.unsqueeze(1), solve_inn).squeeze(-1).squeeze(-1).clamp(0.0, 1e6)
    return {
        "mode": "global_mean",
        "y_mean": y_mean,
        "center_offset": center_offset,
        "center_meas": center_meas,
        "ybar": ybar,
        "innov": innov,
        "innov_kin": innov,
        "R_point": R_point,
        "R_mean": R_mean,
        "R_kin": R_mean,
        "H_mb": H,
        "Cy_mb": Cy_mb,
        "K_mb": K_mb,
        "r_mb": r_mb,
        "Cr_mb": Cr_mb,
        "n_eff": n_eff,
        "nis_mb": nis_mb,
        "h_mean": h_mean,
        "Ch_mix": Ch_mix,
    }


def compute_kinematic_mb_component_batch(
    Z: Tensor,
    mask: Optional[Tensor],
    r_pred: Tensor,
    p_pred: Tensor,
    Cr_pred: Tensor,
    Cp_pred: Tensor,
    Cv: Tensor,
    pi_mu: Tensor,
    h_hat_set: Tensor,
    Ch_set: Tensor,
    r_mean_gamma: float = 0.7,
    min_comp_eff: float = 0.5,
    inactive_R_scale: float = 1e4,
    resp_temp: float = 1.0,
    detach_resp: bool = True,
) -> Dict[str, Tensor]:
    """Component-wise analytic xy update using four GM component batch measurements.

    The old v33 update first collapses all radar points into one global mean.
    This variant keeps one soft mean per GM edge component and stacks the four
    2D means into an 8D Kalman observation for the kinematic state.
    """
    B, _, N = Z.shape
    device, dtype = r_pred.device, r_pred.dtype
    eps = torch.tensor(1e-8, device=device, dtype=dtype)
    H2 = torch.tensor([[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.]], device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
    y_mean, n_eff = masked_mean_points(Z, mask)
    h_mean, Ch_mix = mixture_moments(pi_mu, h_hat_set, Ch_set)
    CI_mix, CII_mix, Sshape = gm_auxiliary_variables(p_pred, Cp_pred, Ch_mix, h_mean)
    center_pred = torch.bmm(H2, r_pred.unsqueeze(-1)).squeeze(-1)
    ybar = center_pred + torch.bmm(Sshape, h_mean.unsqueeze(-1)).squeeze(-1)
    innov = y_mean - ybar
    R_point = make_spd(CI_mix + CII_mix + Cv, eps=1e-7, max_abs=1e5)
    R_mean = make_spd(R_point / n_eff.view(B, 1, 1).clamp_min(1.0).pow(float(r_mean_gamma)), eps=1e-7, max_abs=1e5)

    mu_comp, R_point_comp = [], []
    for j in range(4):
        h_j = h_hat_set[:, :, j]
        Ch_j = Ch_set[:, :, :, j]
        CI_j, CII_j, _ = gm_auxiliary_variables(p_pred, Cp_pred, Ch_j, h_j)
        R_j = make_spd(CI_j + CII_j + Cv, eps=1e-7, max_abs=1e5)
        mu_j = center_pred + torch.bmm(Sshape, h_j.unsqueeze(-1)).squeeze(-1)
        mu_comp.append(mu_j)
        R_point_comp.append(R_j)
    mu_comp_t = torch.stack(mu_comp, dim=1)  # [B,4,2]
    R_point_comp_t = torch.stack(R_point_comp, dim=1)  # [B,4,2,2]

    Z_bn2 = Z.transpose(1, 2)
    if mask is None:
        valid = torch.ones(B, N, device=device, dtype=torch.bool)
    else:
        valid = mask.to(device=device).bool()
    log_pi = torch.log(pi_mu.clamp_min(1e-8))
    logp_all = []
    const = 2.0 * math.log(2.0 * math.pi)
    for j in range(4):
        diff_j = Z_bn2 - mu_comp_t[:, j:j + 1, :]
        R_j = R_point_comp_t[:, j]
        R_rep = R_j.unsqueeze(1).expand(B, N, 2, 2).reshape(B * N, 2, 2)
        diff_rep = diff_j.reshape(B * N, 2, 1)
        sol_j, logdet_j = solve_spd(R_rep, diff_rep, eps=1e-5, max_abs=1e6)
        maha_j = torch.bmm(diff_rep.transpose(1, 2), sol_j).reshape(B, N).clamp(0.0, 100.0)
        logdet_j = logdet_j.reshape(B, N)
        logp_j = log_pi[:, j:j + 1] - 0.5 * (maha_j + logdet_j + const)
        logp_all.append(logp_j)
    logp = torch.stack(logp_all, dim=2)
    logp = torch.where(valid.unsqueeze(2), logp, torch.full_like(logp, -1e9))
    temp = max(float(resp_temp), 1e-4)
    resp = torch.softmax(logp / temp, dim=2) * valid.unsqueeze(2).to(dtype=dtype)
    if detach_resp:
        resp = resp.detach()
    sum_gamma = resp.sum(dim=1)
    sum_gamma_sq = (resp ** 2).sum(dim=1)
    n_eff_comp = (sum_gamma ** 2) / sum_gamma_sq.clamp_min(1e-8)
    y_comp = torch.einsum("bnj,bnd->bjd", resp, Z_bn2) / sum_gamma.clamp_min(1e-8).unsqueeze(-1)
    active = sum_gamma >= float(min_comp_eff)
    y_comp = torch.where(active.unsqueeze(-1), y_comp, mu_comp_t)

    denom_comp = n_eff_comp.clamp_min(1.0).pow(float(r_mean_gamma)).view(B, 4, 1, 1)
    R_comp = R_point_comp_t / denom_comp
    inactive_R = float(inactive_R_scale) * _eye(B, 2, device, dtype).unsqueeze(1)
    R_comp = torch.where(active.view(B, 4, 1, 1), R_comp, inactive_R)
    R_comp = make_spd(R_comp.reshape(B * 4, 2, 2), eps=1e-7, max_abs=1e8).reshape(B, 4, 2, 2)

    y_stack = y_comp.reshape(B, 8)
    ybar_stack = mu_comp_t.reshape(B, 8)
    innov_kin = y_stack - ybar_stack
    H_stack = H2.repeat(1, 4, 1)
    R_stack = torch.zeros(B, 8, 8, device=device, dtype=dtype)
    for j in range(4):
        R_stack[:, 2 * j:2 * j + 2, 2 * j:2 * j + 2] = R_comp[:, j]
    R_stack = make_spd(R_stack, eps=1e-7, max_abs=1e8)

    Cy_mb = make_spd(torch.bmm(torch.bmm(H_stack, Cr_pred), H_stack.transpose(1, 2)) + R_stack, eps=1e-5, max_abs=1e8)
    CrH = torch.bmm(Cr_pred, H_stack.transpose(1, 2))
    K_mb = solve_spd(Cy_mb, CrH.transpose(1, 2), eps=1e-5, max_abs=1e8)[0].transpose(1, 2)
    r_mb = r_pred + torch.bmm(K_mb, innov_kin.unsqueeze(-1)).squeeze(-1)
    I = _eye(B, 5, device, dtype)
    A = I - torch.bmm(K_mb, H_stack)
    Cr_mb = make_spd(torch.bmm(torch.bmm(A, Cr_pred), A.transpose(1, 2)) + torch.bmm(torch.bmm(K_mb, R_stack), K_mb.transpose(1, 2)), eps=1e-6, max_abs=1e8)
    solve_inn, _ = solve_spd(Cy_mb, innov_kin.unsqueeze(-1), eps=1e-5, max_abs=1e8)
    nis_mb = torch.bmm(innov_kin.unsqueeze(1), solve_inn).squeeze(-1).squeeze(-1).clamp(0.0, 1e6)

    comp_center = y_comp - torch.einsum("bij,bkj->bki", Sshape, h_hat_set.transpose(1, 2))
    center_w = torch.where(active, sum_gamma, torch.zeros_like(sum_gamma))
    center_denom = center_w.sum(dim=1, keepdim=True).clamp_min(1e-8)
    center_meas = (comp_center * center_w.unsqueeze(-1)).sum(dim=1) / center_denom
    fallback_center = y_mean - torch.bmm(Sshape, h_mean.unsqueeze(-1)).squeeze(-1)
    center_meas = torch.where((center_w.sum(dim=1, keepdim=True) > 0.0), center_meas, fallback_center)
    center_offset = y_mean - center_meas

    resp_safe = resp.clamp_min(1e-8)
    resp_entropy = (-(resp * resp_safe.log()).sum(dim=2) * valid.to(dtype=dtype)).sum(dim=1) / valid.to(dtype=dtype).sum(dim=1).clamp_min(1.0)
    return {
        "mode": "component_batch",
        "y_mean": y_mean,
        "ybar": ybar,
        "innov": innov,
        "R_point": R_point,
        "R_mean": R_mean,
        "h_mean": h_mean,
        "Ch_mix": Ch_mix,
        "center_meas": center_meas,
        "center_offset": center_offset,
        "H_mb": H_stack,
        "R_kin": R_stack,
        "Cy_mb": Cy_mb,
        "K_mb": K_mb,
        "innov_kin": innov_kin,
        "r_mb": r_mb,
        "Cr_mb": Cr_mb,
        "n_eff": n_eff,
        "nis_mb": nis_mb,
        "comp_n_eff": n_eff_comp,
        "comp_sum_gamma": sum_gamma,
        "comp_active": active.to(dtype=dtype),
        "resp_entropy": resp_entropy,
        "y_comp_mean": y_comp,
        "y_comp_pred": mu_comp_t,
    }


def compute_kinematic_mb(
    Z: Tensor,
    mask: Optional[Tensor],
    r_pred: Tensor,
    p_pred: Tensor,
    Cr_pred: Tensor,
    Cp_pred: Tensor,
    Cv: Tensor,
    pi_mu: Tensor,
    h_hat_set: Tensor,
    Ch_set: Tensor,
    r_mean_gamma: float = 0.7,
    kin_update_mode: str = "global_mean",
    min_comp_eff: float = 0.5,
    inactive_R_scale: float = 1e4,
    resp_temp: float = 1.0,
    detach_resp: bool = True,
) -> Dict[str, Tensor]:
    if kin_update_mode == "global_mean":
        return compute_kinematic_mb_global_mean(
            Z, mask, r_pred, p_pred, Cr_pred, Cp_pred, Cv,
            pi_mu, h_hat_set, Ch_set, r_mean_gamma=r_mean_gamma,
        )
    if kin_update_mode == "component_batch":
        return compute_kinematic_mb_component_batch(
            Z, mask, r_pred, p_pred, Cr_pred, Cp_pred, Cv,
            pi_mu, h_hat_set, Ch_set,
            r_mean_gamma=r_mean_gamma,
            min_comp_eff=min_comp_eff,
            inactive_R_scale=inactive_R_scale,
            resp_temp=resp_temp,
            detach_resp=detach_resp,
        )
    raise ValueError(f"unknown kin_update_mode={kin_update_mode}")


class KNetMEMBackend(nn.Module):
    """Small deterministic MEM filter shell used by the v33 training loop."""

    def __init__(self, dt: float = 0.1, motion_model: str = "cv"):
        super().__init__()
        self.dt = float(dt)
        if motion_model not in {"cv", "ct"}:
            raise ValueError(f"unknown motion_model={motion_model}")
        self.motion_model = motion_model

    @staticmethod
    def _ct_terms(theta: Tensor, dt: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Stable A, B and omega derivatives for exact coordinated turn."""
        small = theta.abs() < 1e-4
        z2 = theta * theta
        sinc_series = 1.0 - z2 / 6.0 + z2 * z2 / 120.0
        cosc_series = theta / 2.0 - theta * z2 / 24.0 + theta * z2 * z2 / 720.0
        dsinc_series = -theta / 3.0 + theta * z2 / 30.0 - theta * z2 * z2 / 840.0
        dcosc_series = 0.5 - z2 / 8.0 + z2 * z2 / 144.0
        safe_z = torch.where(small, torch.ones_like(theta), theta)
        sinc = torch.where(small, sinc_series, torch.sin(theta) / safe_z)
        cosc = torch.where(small, cosc_series, (1.0 - torch.cos(theta)) / safe_z)
        dsinc = torch.where(
            small, dsinc_series,
            (theta * torch.cos(theta) - torch.sin(theta)) / (safe_z * safe_z),
        )
        dcosc = torch.where(
            small, dcosc_series,
            (theta * torch.sin(theta) - (1.0 - torch.cos(theta))) / (safe_z * safe_z),
        )
        return dt * sinc, dt * cosc, dt * dt * dsinc, dt * dt * dcosc

    def predict(
        self,
        r: Tensor,
        p: Tensor,
        Cr: Tensor,
        Cp: Tensor,
        Cwr: Tensor,
        Cwp: Tensor,
        dt: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        B = r.shape[0]
        device, dtype = r.device, r.dtype
        if dt is None:
            dt_b = torch.full((B,), self.dt, device=device, dtype=dtype)
        else:
            dt_b = torch.as_tensor(dt, device=device, dtype=dtype)
            if dt_b.ndim == 0:
                dt_b = dt_b.expand(B)
            elif dt_b.ndim == 1 and dt_b.numel() == B:
                dt_b = dt_b.reshape(B)
            else:
                raise ValueError(f"dt must be scalar or shape [B], got {tuple(dt_b.shape)}")
        r_pred = r.clone()
        p_pred = p.clone()
        # _eye uses expand; clone before per-sample Jacobian entries are written.
        Ar = _eye(B, 5, device, dtype).clone()
        if self.motion_model == "cv":
            r_pred[:, 0] = r[:, 0] + r[:, 2] * dt_b
            r_pred[:, 1] = r[:, 1] + r[:, 3] * dt_b
            Ar[:, 0, 2] = dt_b
            Ar[:, 1, 3] = dt_b
        else:
            omega, vx, vy = r[:, 4], r[:, 2], r[:, 3]
            theta = omega * dt_b
            s, c = torch.sin(theta), torch.cos(theta)
            A, Bturn, dA, dB = self._ct_terms(theta, dt_b)
            r_pred[:, 0] = r[:, 0] + A * vx - Bturn * vy
            r_pred[:, 1] = r[:, 1] + Bturn * vx + A * vy
            r_pred[:, 2] = c * vx - s * vy
            r_pred[:, 3] = s * vx + c * vy
            Ar[:, 0, 2] = A
            Ar[:, 0, 3] = -Bturn
            Ar[:, 1, 2] = Bturn
            Ar[:, 1, 3] = A
            Ar[:, 0, 4] = dA * vx - dB * vy
            Ar[:, 1, 4] = dB * vx + dA * vy
            Ar[:, 2, 2] = c
            Ar[:, 2, 3] = -s
            Ar[:, 3, 2] = s
            Ar[:, 3, 3] = c
            Ar[:, 2, 4] = -dt_b * (s * vx + c * vy)
            Ar[:, 3, 4] = dt_b * (c * vx - s * vy)
        p_pred[:, 0] = wrap_to_pi(p[:, 0] + r[:, 4] * dt_b)
        Ap = _eye(B, 3, device, dtype)
        Cr_pred = make_spd(torch.bmm(torch.bmm(Ar, Cr), Ar.transpose(1, 2)) + Cwr, eps=1e-6, max_abs=1e6)
        Cp_raw = torch.bmm(torch.bmm(Ap, Cp), Ap.transpose(1, 2)) + Cwp
        if self.motion_model == "ct":
            # Split covariance has no yaw/omega cross block. Project the prior
            # omega uncertainty once; Cwp remains the independent shape noise.
            Cp_raw = Cp_raw.clone()
            Cp_raw[:, 0, 0] = Cp_raw[:, 0, 0] + dt_b.square() * Cr[:, 4, 4]
        Cp_pred = make_spd(Cp_raw, eps=1e-7, max_abs=1e4)
        return r_pred, p_pred, Cr_pred, Cp_pred

    def joseph_update_general(self, r_pred: Tensor, Cr_pred: Tensor, K: Tensor, H: Tensor, innov: Tensor, R: Tensor) -> Tuple[Tensor, Tensor]:
        B = r_pred.shape[0]
        device, dtype = r_pred.device, r_pred.dtype
        r_plus = r_pred + torch.bmm(K, innov.unsqueeze(-1)).squeeze(-1)
        I = _eye(B, 5, device, dtype)
        A = I - torch.bmm(K, H)
        Cr_plus = make_spd(torch.bmm(torch.bmm(A, Cr_pred), A.transpose(1, 2)) + torch.bmm(torch.bmm(K, R), K.transpose(1, 2)), eps=1e-6, max_abs=1e8)
        return r_plus, Cr_plus

    def joseph_update(self, r_pred: Tensor, Cr_pred: Tensor, K: Tensor, innov: Tensor, R_mean: Tensor) -> Tuple[Tensor, Tensor]:
        B = r_pred.shape[0]
        device, dtype = r_pred.device, r_pred.dtype
        H = torch.tensor([[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.]], device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
        return self.joseph_update_general(r_pred, Cr_pred, K, H, innov, R_mean)
