# E129 common module: Causal (t-1, t-3) proprioception extension (164->230D) and WAMv1 / WAMv2 architectures.
import os, sys, math, json, hashlib, datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.update(
    E102_GROUP="A2",
    PYTHONDONTWRITEBYTECODE="1",
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    OMP_NUM_THREADS="4",
)

T = Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu")
for p in ("code_e110f", "code_e106c/run02", "code_e106c", "code_e102a"):
    sys.path.insert(0, str(T / p))

from e99w2_common import GROUP
assert GROUP == "A2"
from e103d_model import WorldModel as BaseE103Model
from e99w2_model import JOINTS, trajectory
from e102a_contract import contract, fixed_np, states as grip_states
from e88.model import horizon_embedding, directional_mask

INIT = T / "results/e106c_f2_s90_20261115_02/ckpt_20000.pt"
D_CKPT = T / "results/e110f_d_20261116_05/ckpt_04000.pt"
LOCK_PATH = T / "results/e106c_control_20261115_02/PRETRAIN_LOCK.json"
CTRL = T / "results/e129_control_20261119_01"
GR = [6, 13]
SHORT_OFFSETS = (-3, -1)  # 100ms and 33ms causal instantaneous velocity!


def brt():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat()


def extra_causal_66(q, pose, a, z, good, c):
    # Strictly causal check matching PlannedData.get: same segment, valid state, available_time <= anchor
    cur = fixed_np(q[a], c)
    p0 = pose[a]
    g0 = grip_states(q[a], c)
    dq_list, dp_list, flags = [], [], []
    seg = z["continuous_segment"]
    avail_t = z["state_available_time"]
    for off in SHORT_OFFSETS:
        j = a + off
        ok = bool(
            j >= 0
            and good[j]
            and seg[j] == seg[a]
            and avail_t[j] <= avail_t[a] + 1e-7
        )
        flags.append(float(ok))
        if ok:
            dq = fixed_np(q[j], c) - cur
            dq[GR] = grip_states(q[j], c) - g0
            dp = pose[j] - p0
        else:
            dq = np.zeros(14, dtype="f4")
            dp = np.zeros(18, dtype="f4")
        dq_list.append(dq)
        dp_list.append(dp)
    vec = np.concatenate([*dq_list, *dp_list, np.asarray(flags, dtype="f4")]).astype("f4")
    assert vec.shape == (66,) and np.isfinite(vec).all()
    return vec


def build_or_load_extra_stats(tr, dev, dev_entries):
    assert "piper_4dc107335389f5711a0f65e0" not in {r["canonical_id"] for r in tr.rows}, "episode142 leaked into train!"
    assert "piper_4dc107335389f5711a0f65e0" not in {e["cid"] for e in dev_entries}, "episode142 leaked into dev!"
    CTRL.mkdir(parents=True, exist_ok=True)
    stats_file = CTRL / "extra_proprio_stats.json"
    c = contract()
    raw_tr = np.empty((len(tr.rows), 66), dtype="f4")
    for i, r in enumerate(tr.rows):
        z, q, pose, good = tr.states.get(tr, r["canonical_id"])
        raw_tr[i] = extra_causal_66(q, pose, r["anchor"], z, good, c)

    if stats_file.exists():
        st = json.loads(stats_file.read_text())
        mean66 = np.asarray(st["mean66"], dtype="f4")
        std66 = np.asarray(st["std66"], dtype="f4")
    else:
        mean66 = raw_tr.mean(0).astype("f4")
        std66 = np.maximum(raw_tr.std(0), 0.01).astype("f4")
        st = dict(
            fitted_split="train",
            train_rows=len(tr.rows),
            offsets=list(SHORT_OFFSETS),
            dims=66,
            valid_rate_m3=float(raw_tr[:, 64].mean()),
            valid_rate_m1=float(raw_tr[:, 65].mean()),
            mean66=mean66.tolist(),
            std66=std66.tolist(),
            created_brt=brt(),
        )
        tmp = stats_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=2) + "\n")
        os.replace(tmp, stats_file)

    norm_tr = ((raw_tr - mean66) / std66).astype("f4")
    norm_dev = {}
    for e in dev_entries:
        z, q, pose, good = dev.states.get(dev, e["cid"])
        v = extra_causal_66(q, pose, e["anchor"], z, good, c)
        norm_dev[e["index"]] = ((v - mean66) / std66).astype("f4")
    return norm_tr, norm_dev


def attach_extra_proprio(b, extra_rows_np):
    ex = torch.from_numpy(np.asarray(extra_rows_np, dtype="f4")).to(b["proprio"].device, b["proprio"].dtype)
    out = dict(b)
    out["proprio"] = torch.cat([b["proprio"], ex], dim=1)
    return out


class UpgradedWAM(BaseE103Model):
    """
    variant='v1': 230-D proprio_proj (with t-1, t-3 causal velocity) -> state token.
    variant='v2': v1 + zero-initialized kinematic query highway (230 -> 512 -> 50x256)
                  injected into the 50 action query tokens BEFORE the 6 Transformer layers,
                  enabling per-step velocity-seeded cross-attention to 3-camera 2x4 visual tokens.
    """
    def __init__(self, norm, variant="v2"):
        super().__init__(norm)
        assert variant in ("v1", "v2")
        self.e129_variant = variant
        if variant == "v2":
            self.kin_mlp = nn.Sequential(
                nn.Linear(230, self.c.width),
                nn.LayerNorm(self.c.width),
                nn.GELU(),
                nn.Linear(self.c.width, self.c.width),
            )
            self.kin_time_basis = nn.Parameter(torch.zeros(50, self.c.width))
        else:
            self.kin_mlp = None
            self.kin_time_basis = None

    def widen_from_164(self):
        old = self.proprio_proj[0]
        assert old.in_features == 164
        new = nn.Linear(230, old.out_features, device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            new.weight[:, :164].copy_(old.weight)
            new.weight[:, 164:].zero_()
            new.bias.copy_(old.bias)
        self.proprio_proj[0] = new
        return self

    def forward(self, b, a=None, w=None, ta=None, tw=None, branch="mean"):
        if b.get("external_no_robot", False):
            return super().forward(b, a, w, ta, tw, branch)
        if self.kin_mlp is None:
            return super().forward(b, a, w, ta, tw, branch)

        # Exact BaseE103Model / e99w2_model forward, adding rank-256 velocity-seeded basis onto the 50 action queries `at`
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
            if self.training:
                state = state + torch.randn_like(state) * self.c.state_noise
                sv = sv & (torch.rand(len(o), device=o.device) >= self.c.state_drop)
            st = (self.state_proj(state * sv[:, None])[:, None] + self.kind[1]) * sv[:, None, None]
            st = st + self.proprio_proj(b["proprio"])[:, None]
            kin_q = self.kin_mlp(b["proprio"])[:, None, :] * self.kin_time_basis[None, :, :]
            at = (
                self.action_proj(a)
                + self.a_pos[None]
                + self.ta(torch.stack([ta, torch.sin(ta * math.pi)], 1))[:, None]
                + self.kind[1]
                + kin_q
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
        ah = x[:, no + 1 : no + 51] if na else None
        wh = x[:, -nw:] if nw else None

        def mean_fn(z, v):
            return (z * v[:, :, None]).sum(1) / v.sum(1, keepdim=True).clamp_min(1)

        image = mean_fn(x[:, :24], ov[:, :24])
        hist = mean_fn(x[:, 24:56], ov[:, 24:56])
        causal = mean_fn(x[:, :no], ov)
        clean = causal[:, None] + he + pos[None] if nw else None
        aux = clean.mean(1) if nw else causal * 0
        reps = [image, hist, ah.mean(1) if na else image * 0]
        return dict(
            grip_logits=self.grip_head(ah) if na else None,
            delta=self.delta_head(ah) if na else None,
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
