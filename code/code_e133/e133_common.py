# E133 common module:
#   - Differentiable 3D fingertip position + orientation + velocity-slope loss (G1)
#   - Transition-weighted gripper BCE + dedicated switch-window & switch-frame CDF head (G2)
#   - Training-time visual spatial token dropout (15% / 20%) + optional kin_skip (G3 / G4)
import os, sys, math, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

T = Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu")
for sub in ("code_e130", "code_e129"):
    if str(T / sub) not in sys.path:
        sys.path.insert(0, str(T / sub))

from e129_common import (
    INIT, D_CKPT, LOCK_PATH, GR, SHORT_OFFSETS, brt,
    UpgradedWAM, build_or_load_extra_stats, attach_extra_proprio,
)
from e99w2_model import JOINTS, trajectory, pose_torch
from e88.model import horizon_embedding, directional_mask

CTRL133 = T / "results/e133_control_20261119_01"
POS_IDX = [0, 1, 2, 9, 10, 11]
ROT_IDX = [3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17]


class E133WAM(UpgradedWAM):
    def __init__(self, norm, use_switch_head=False, vis_token_drop=0.0, use_kin_skip=False):
        super().__init__(norm, variant="v1")
        self.use_switch_head = bool(use_switch_head)
        self.vis_token_drop = float(vis_token_drop)
        self.use_kin_skip = bool(use_kin_skip)

        if self.use_switch_head:
            self.switch_win_head = nn.Linear(self.c.width * 2, 4)
            self.switch_step_head = nn.Linear(self.c.width, 4)
            self.switch_to_grip = nn.Linear(4, 2)
            nn.init.zeros_(self.switch_to_grip.weight)
            nn.init.zeros_(self.switch_to_grip.bias)
        else:
            self.switch_win_head = None
            self.switch_step_head = None
            self.switch_to_grip = None

        if self.use_kin_skip:
            self.kin_skip = nn.Sequential(
                nn.Linear(14 + 230, 512),
                nn.GELU(),
                nn.Linear(512, 512),
                nn.GELU(),
                nn.Linear(512, 50 * 12),
            )
            nn.init.zeros_(self.kin_skip[-1].weight)
            nn.init.zeros_(self.kin_skip[-1].bias)
        else:
            self.kin_skip = None

    def forward(self, b, a=None, w=None, ta=None, tw=None, branch="mean"):
        if b.get("external_no_robot", False):
            return super().forward(b, a, w, ta, tw, branch)

        o, ov, hv = self.observation(b)
        if self.training and (a is not None) and (w is None) and self.vis_token_drop > 0.0:
            keep = torch.rand((len(o), 24), device=o.device) >= self.vis_token_drop
            too_few = (ov[:, :24] & keep).sum(dim=1) < 4
            if too_few.any():
                keep[too_few] = True
            ov = ov.clone()
            ov[:, :24] = ov[:, :24] & keep

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
        sw_win_logits = None
        sw_step_logits = None

        if na:
            delta_out = self.delta_head(ah)
            if self.kin_skip is not None:
                kin_in = torch.cat([b["q_normalized"] * b["state_valid"][:, None], b["proprio"]], dim=-1)
                delta_out = delta_out + self.kin_skip(kin_in).reshape(-1, 50, 12)

            grip_out = self.grip_head(ah)
            if self.use_switch_head:
                sw_win_logits = self.switch_win_head(torch.cat([st_out, ah.mean(1)], dim=-1))
                sw_step_logits = self.switch_step_head(ah)
                p_win = torch.sigmoid(sw_win_logits)
                cdf_step = torch.cumsum(torch.softmax(sw_step_logits, dim=1), dim=1)
                sw_feat = p_win[:, None, :] * cdf_step
                grip_out = grip_out + self.switch_to_grip(sw_feat)

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
            switch_win_logits=sw_win_logits,
            switch_step_logits=sw_step_logits,
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
