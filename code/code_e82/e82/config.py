from __future__ import annotations
from dataclasses import dataclass, asdict, fields
from pathlib import Path
import hashlib, json

@dataclass(frozen=True)
class Config:
    schema_version: str = 'e82.piper.v1'
    robot: str = 'piper'
    seed: int = 8101
    width: int = 256
    layers: int = 6
    heads: int = 8
    ffn: int = 1024
    latent_dim: int = 768
    action_dim: int = 14
    horizon: int = 50
    hz: int = 30
    tasks: int = 2
    lora_rank: int = 8
    batch_size: int = 32
    microbatch: int = 1
    euler_steps: int = 10
    noise_seeds: tuple = (47041, 47042, 47043, 47044)
    p1_steps: int = 5000
    warm_steps: int = 1000
    joint_steps: int = 9000
    lambda_world: float = 0.1
    lambda_ramp: int = 500
    budget_gpuh: float = 48.0
    profile_budget_gpuh: float = 0.4
    cache_gb: float = 100.0
    dino_weights: str = '/mnt/workspace/dengjianfeng/models/dinov2_official_20261031/dinov2_vitb14_pretrain.pth'
    data_root: str = '/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e82_piper_v1'
    def __post_init__(self):
        assert all(isinstance(getattr(self,k),int) and getattr(self,k)>0 for k in ('seed','batch_size','microbatch','p1_steps','warm_steps','joint_steps'))
        assert 4<self.budget_gpuh<=48 and self.lambda_world==.1 and self.lambda_ramp==500
        assert self.robot == 'piper', 'Tianji adapter reserved, not implemented'
        assert (self.width,self.layers,self.heads,self.ffn)==(256,6,8,1024)
        assert (self.latent_dim,self.action_dim,self.horizon,self.hz)==(768,14,50,30)
        assert self.lora_rank==8 and self.tasks>=2
        assert len(self.noise_seeds)==4 and self.euler_steps==10
        assert self.batch_size % self.microbatch == 0 and self.microbatch>0
        assert self.cache_gb<=100 and 0<self.profile_budget_gpuh<1
    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()
    @classmethod
    def load(cls,path):
        d=json.loads(Path(path).read_text())
        extra=set(d)-{f.name for f in fields(cls)}
        if extra: raise ValueError(f'Unknown config keys: {extra}')
        return cls(**d)

TOKEN_TABLE={'top':25,'left_full':17,'left_roi':16,'right_full':17,'right_roi':16,
             'visual_per_time':91,'observation_visual':182,'state':4,'task':1,
             'O':187,'A':50,'W':273,'joint':510,'action_only':237,'P1':460}
GROUPS=((0,25,1/3),(25,42,1/6),(42,58,1/6),(58,75,1/6),(75,91,1/6))
CAMERA_IDS=([0]*25+[1]*33+[2]*33)
ENDPOINTS=(10,25,50)
