# E134 common module:
#   1. DecoupledGripperSwitch: gradient-detached from shared 6-layer backbone + 20% gripper-only dropout + 1D-Conv temporal adapter
#   2. TemporalFlowHead: 4-block AdaLN-Zero Temporal DiT + Depthwise-Conv1D velocity network for 50-step Flow Matching
#   3. Smooth AR(1) Gaussian trajectory prior (rho=0.85) + 8-step Midpoint/Euler ODE sampler (zero / stochastic / mean8 / best8)
import os, sys, math, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

T = Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu")
for sub in ("code_e133", "code_e132", "code_e131", "code_e130", "code_e129"):
    if str(T / sub) not in sys.path:
        sys.path.insert(0, str(T / sub))

from e129_common import (
    INIT, D_CKPT, LOCK_PATH, GR, SHORT_OFFSETS, brt,
    UpgradedWAM, build_or_load_extra_stats, attach_extra_proprio,
)
from e99w2_model import JOINTS, trajectory, pose_torch
from e102a_contract import contract
from e88.model import horizon_embedding, directional_mask

CTRL134 = T / "results/e134_control_20261119_02"
G1_CKPT = T / "results/e133_wam_g1_s90_20261119_01/ckpt_04000.pt"
POS_IDX = [0, 1, 2, 9, 10, 11]
ROT_IDX = [3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17]


def sample_smooth_noise(shape, device, dtype=torch.float32, rho=0.85, generator=None):
    """
    Stationary AR(1) trajectory noise (B, 50, D): unit marginal variance at every step t,
    temporally smooth correlation rho=0.85 across adjacent steps.
    """
    b, steps, d = shape
    raw = torch.randn((b, steps, d), device=device, dtype=dtype, generator=generator)
    if rho <= 0.0:
        return raw
    scale = math.sqrt(1.0 - rho * rho)
    cols = [raw[:, 0]]
    for t in range(1, steps):
        cols.append(rho * cols[-1] + scale * raw[:, t])
    return torch.stack(cols, dim=1)


class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
        args = t[:, None] * freqs[None, :] * 1000.0
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class TemporalFlowBlock(nn.Module):
    def __init__(self, width=256, heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False)
        self.dwconv = nn.Conv1d(width, width, kernel_size=5, padding=2, groups=width)
        self.ffn = nn.Sequential(nn.Linear(width, width * 3), nn.GELU(), nn.Linear(width * 3, width))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(width, width * 6))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, g_cond):
        s1, b1, g1, s2, b2, g2 = self.ada(g_cond)[:, None, :].chunk(6, dim=-1)
        h1 = self.norm1(x) * (1.0 + s1) + b1
        a_out, _ = self.attn(h1, h1, h1, need_weights=False)
        x = x + g1 * a_out

        h2 = self.norm2(x) * (1.0 + s2) + b2
        h2 = h2 + self.dwconv(h2.transpose(1, 2)).transpose(1, 2)
        x = x + g2 * self.ffn(h2)
        return x


class TemporalFlowHead(nn.Module):
    """
    4-block AdaLN-Zero Temporal DiT + Conv1D velocity field v_theta(x_t, t | ah, st_out).
    """
    def __init__(self, act_dim=12, width=256, depth=4, heads=8):
        super().__init__()
        self.in_proj = nn.Linear(act_dim, width)
        self.cond_proj = nn.Linear(width, width)
        self.st_proj = nn.Linear(width, width)
        self.pos_emb = nn.Parameter(torch.randn(1, 50, width) * 0.02)
        self.time_emb = SinusoidalTimeEmb(width)
        self.blocks = nn.ModuleList([TemporalFlowBlock(width, heads) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(width, width * 2))
        self.out_proj = nn.Linear(width, act_dim)
        nn.init.zeros_(self.final_ada[-1].weight)
        nn.init.zeros_(self.final_ada[-1].bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x_t, t, ah_cond, st_cond):
        g = self.time_emb(t) + self.st_proj(st_cond)
        x = self.in_proj(x_t) + self.cond_proj(ah_cond) + self.pos_emb
        for blk in self.blocks:
            x = blk(x, g)
        s, b = self.final_ada(g)[:, None, :].chunk(2, dim=-1)
        x = self.final_norm(x) * (1.0 + s) + b
        return self.out_proj(x)


class DecoupledGripperSwitch(nn.Module):
    """
    Reads gradient-detached backbone tokens (st_out.detach(), ah.detach()) with 20% gripper-only dropout
    + 1D temporal conv adapter + switch-window & switch-frame CDF head. Zero gradient interference on arm backbone!
    """
    def __init__(self, width=256, drop=0.20):
        super().__init__()
        self.drop = nn.Dropout(drop)
        self.conv = nn.Conv1d(width, width, kernel_size=5, padding=2)
        self.refine = nn.Sequential(nn.LayerNorm(width), nn.GELU(), nn.Linear(width, width))
        self.res_grip = nn.Linear(width, 2)
        self.switch_win_head = nn.Linear(width * 2, 4)
        self.switch_step_head = nn.Linear(width, 4)
        self.switch_to_grip = nn.Linear(4, 2)
        nn.init.zeros_(self.res_grip.weight)
        nn.init.zeros_(self.res_grip.bias)
        nn.init.zeros_(self.switch_to_grip.weight)
        nn.init.zeros_(self.switch_to_grip.bias)

    def forward(self, base_grip_logits, st_out, ah):
        st_d = self.drop(st_out.detach())
        ah_d = self.drop(ah.detach())
        h = ah_d + self.refine(self.conv(ah_d.transpose(1, 2)).transpose(1, 2))
        sw_win = self.switch_win_head(torch.cat([st_d, h.mean(1)], dim=-1))
        sw_step = self.switch_step_head(h)
        p_win = torch.sigmoid(sw_win)
        cdf = torch.cumsum(torch.softmax(sw_step, dim=1), dim=1)
        grip = base_grip_logits.detach() + self.res_grip(h) + self.switch_to_grip(p_win[:, None, :] * cdf)
        return grip, sw_win, sw_step


class E134WAM(UpgradedWAM):
    """
    arm_mode:
      - 'det_decoupled': deterministic delta_head + decoupled gripper switch
      - 'fm_native':     native 6-layer Transformer Flow Matching (action_proj + ta + action_head) + decoupled gripper
      - 'fm_dit':        cached-backbone TemporalFlowHead (pure conditional FM) + decoupled gripper
      - 'fm_residual':   base delta_head mean + TemporalFlowHead residual FM + decoupled gripper
    """
    def __init__(self, norm, arm_mode="fm_dit"):
        super().__init__(norm, variant="v1")
        assert arm_mode in ("det_decoupled", "fm_native", "fm_dit", "fm_residual")
        self.arm_mode = arm_mode
        self.dec_grip = DecoupledGripperSwitch(self.c.width, drop=0.20)
        if arm_mode in ("fm_dit", "fm_residual"):
            self.flow_head = TemporalFlowHead(act_dim=12, width=self.c.width, depth=4, heads=8)
        else:
            self.flow_head = None

    def forward(self, b, a=None, w=None, ta=None, tw=None, branch="mean"):
        if b.get("external_no_robot", False):
            return super().forward(b, a, w, ta, tw, branch)

        o, ov, hv = self.observation(b)
        no = o.shape[1]
        parts = [o]
        valids = [ov]
        na = 0
        nw = 0
        nr = 0
        if a is not None:
            state = b["q_normalized"].clone()
            sv = b["state_valid"].clone()
            st = (self.state_proj(state * sv[:, None])[:, None] + self.kind[1]) * sv[:, None, None]
            st = st + self.proprio_proj(b["proprio"])[:, None]
            at = (
                self.action_proj(a)
                + self.a_pos[None]
                + self.ta(torch.stack([ta, torch.sin(ta * math.pi)], 1))[:, None]
                + self.kind[1]
            )
            parts.extend([st, at])
            valids.extend([b["state_valid"][:, None], torch.ones(a.shape[:2], dtype=torch.bool, device=o.device)])
            na = 51
        if w is not None:
            hs = b["world_horizons"]
            assert torch.equal(hs, hs[0].expand_as(hs))
            nw = w.shape[1]
            he = self.horizon_proj(horizon_embedding(hs.to(o.dtype), self.c.width))
            pos = self.grid_pos[b["world_positions"]]
            if a is not None and not b.get("fixed_condition", False):
                absolute = a * b["action_std"] + b["action_mean"] + b["q_anchor"][:, None]
                robot = trajectory(b["context"], absolute)
            else:
                robot = b["robot"]
            rh = torch.arange(101, device=o.device)
            r = self.robot_proj((robot - self.robot_mean) / self.robot_scale) + self.horizon_proj(horizon_embedding(rh.to(o.dtype), self.c.width))[None] + self.kind[1]
            query = self.mean_query.expand(len(o), nw, -1) + he + pos[None] + self.kind[2]
            parts.extend([r, query])
            valids.extend([b["robot_valid"], b["world_valid"]])
            nr = 101

        x = torch.cat(parts, 1)
        n = len(x[0])
        m = torch.ones(n, n, dtype=torch.bool, device=o.device)
        m[:no, :no] = False
        if na:
            m[no : no + na, : no + na] = False
        if nw:
            rs = no + na
            ws = rs + nr
            m[rs:ws, rs:ws] = ~torch.eye(nr, dtype=torch.bool, device=o.device)
            m[ws:, : no + na] = False
            m[ws:, rs:ws] = rh[None, :] > hs[0, :, None]
            m[ws:, ws:] = directional_mask(no, 0, hs[0])[no:, no:]
            if na:
                m[no : no + na, ws:] = False
                m[ws:, :][hs[0] > 50, no : no + na] = True

        invalid = ~torch.cat(valids, 1)
        fast = torch.backends.mha.get_fastpath_enabled()
        if not self.training:
            torch.backends.mha.set_fastpath_enabled(False)
        try:
            for block in self.blocks:
                x = block(x, m, invalid)
        finally:
            if not self.training:
                torch.backends.mha.set_fastpath_enabled(fast)

        x = self.norm(x)
        st_out = x[:, no] if na else None
        ah = x[:, no + 1 : no + 51] if na else None
        wh = x[:, -nw:] if nw else None

        delta_out = None
        grip_out = None
        sw_win = None
        sw_step = None
        if na:
            delta_out = self.delta_head(ah)
            base_g = self.grip_head(ah)
            grip_out, sw_win, sw_step = self.dec_grip(base_g, st_out, ah)

        def mean_fn(z, v):
            return (z * v[:, :, None]).sum(1) / v.sum(1, keepdim=True).clamp_min(1)

        image = mean_fn(x[:, :24], ov[:, :24])
        hist = mean_fn(x[:, 24:56], ov[:, 24:56])
        causal = mean_fn(x[:, :no], ov)
        clean = causal[:, None] + he + pos[None] if nw else None
        aux = clean.mean(1) if nw else causal * 0
        reps = [image, hist, ah.mean(1) if na else image * 0]
        return dict(
            grip_logits=grip_out,
            switch_win_logits=sw_win,
            switch_step_logits=sw_step,
            st_out=st_out,
            ah=ah,
            delta=delta_out,
            mean=self.mean_head(wh) if nw else None,
            action=self.action_head(ah) if na else None,
            world=None,
            world_cos=self.world_cos_head(clean) if nw else None,
            observation=x[:, :no],
            phase=self.phase_head(aux),
            progress=self.progress_head(aux).sigmoid().squeeze(-1),
            align=[F.normalize(h(z), dim=-1) for h, z in zip(self.align_heads, reps)],
            history_kept=hv,
            alignment_available=torch.stack(
                [b["current_valid"].any(1), hv.any(1), torch.full((len(o),), bool(na), device=o.device, dtype=torch.bool)],
                1,
            ),
        )

    @torch.no_grad()
    def infer_e134(self, b, steps=8, sample_mode="stochastic", n_samples=8, seed=90):
        """
        Unified inference returning absolute (B, 50, 14) action (or (K, B, 50, 14) when sample_mode='all_k').
        sample_mode: 'stochastic' (N=1 sample), 'zero' (x0=0 deterministic ODE), 'mean8' (mean of 8 samples), 'all_k'
        """
        device = b["obs"].device
        bsz = len(b["obs"])
        t0 = torch.zeros(bsz, device=device)
        a0 = torch.zeros((bsz, 50, 14), device=device, dtype=b["q_current"].dtype)
        heads = self(b, a0, None, t0, t0)
        grip_logits = heads["grip_logits"]
        c_dict = contract()
        j_idx = JOINTS

        def to_abs(delta_12):
            act = b["q_current"][:, None].expand(-1, 50, -1).clone()
            act[:, :, j_idx] = b["q_current"][:, None, j_idx] + delta_12 * b["action_std"][:, :, j_idx]
            act[:, :, [6, 13]] = torch.where(
                grip_logits > 0,
                act.new_tensor(c_dict["open_m"]),
                act.new_tensor(c_dict["closed_m"]),
            )
            return act

        if self.arm_mode == "det_decoupled":
            act = to_abs(heads["delta"])
            return act[None].expand(n_samples, -1, -1, -1) if sample_mode == "all_k" else act

        gen = torch.Generator(device=device).manual_seed(seed)

        if self.arm_mode == "fm_native":
            K = n_samples if sample_mode in ("mean8", "all_k") else 1
            if sample_mode == "zero":
                x = torch.zeros((K * bsz, 50, 12), device=device, dtype=a0.dtype)
            else:
                x = sample_smooth_noise((K * bsz, 50, 12), device, dtype=a0.dtype, rho=0.85, generator=gen)
            # Expand batch by K if K > 1
            b_exp = {k: (v.repeat(K, *([1] * (v.ndim - 1))) if torch.is_tensor(v) and v.ndim > 0 and len(v) == bsz else v) for k, v in b.items()}
            dt = 1.0 / steps
            for s_i in range(steps):
                t_cur = torch.full((K * bsz,), (s_i + 0.5) * dt, device=device, dtype=a0.dtype)
                a_in = torch.zeros((K * bsz, 50, 14), device=device, dtype=a0.dtype)
                a_in[:, :, j_idx] = x
                v = self(b_exp, a_in, None, t_cur, t_cur)["action"][:, :, j_idx]
                x = x + v * dt
            x_k = x.reshape(K, bsz, 50, 12)
            if sample_mode == "all_k":
                return torch.stack([to_abs(x_k[i]) for i in range(K)], dim=0)
            return to_abs(x_k.mean(0))

        # Cached-backbone TemporalFlowHead ('fm_dit' or 'fm_residual')
        ah_c = heads["ah"]
        st_c = heads["st_out"]
        base_mu = heads["delta"] if self.arm_mode == "fm_residual" else torch.zeros_like(heads["delta"])
        K = n_samples if sample_mode in ("mean8", "all_k") else 1
        if sample_mode == "zero":
            x = torch.zeros((K * bsz, 50, 12), device=device, dtype=a0.dtype)
        else:
            # For fm_residual, residual std is ~0.45 of full target_delta std; AR1(rho=0.85) handles any target cleanly!
            x = sample_smooth_noise((K * bsz, 50, 12), device, dtype=a0.dtype, rho=0.85, generator=gen)

        ah_exp = ah_c.repeat(K, 1, 1)
        st_exp = st_c.repeat(K, 1)
        dt = 1.0 / steps
        # 2nd-order Midpoint ODE solver (8 steps = 16 tiny-head evals, ~1.2ms total!)
        for s_i in range(steps):
            t_0 = torch.full((K * bsz,), s_i * dt, device=device, dtype=a0.dtype)
            t_mid = torch.full((K * bsz,), (s_i + 0.5) * dt, device=device, dtype=a0.dtype)
            v1 = self.flow_head(x, t_0, ah_exp, st_exp)
            x_mid = x + 0.5 * dt * v1
            v_mid = self.flow_head(x_mid, t_mid, ah_exp, st_exp)
            x = x + dt * v_mid

        x_k = base_mu[None] + x.reshape(K, bsz, 50, 12)
        if sample_mode == "all_k":
            return torch.stack([to_abs(x_k[i]) for i in range(K)], dim=0)
        return to_abs(x_k.mean(0))
