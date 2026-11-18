"""Reserved extension contracts. E82 ships only Piper; no Tianji or legacy oracle code."""
from typing import Protocol,Mapping,Any

class RobotAdapter(Protocol):
    robot: str
    action_dim: int
    horizon: int
    def observation(self,batch:Mapping[str,Any])->tuple: ...
    def decode_action(self,normalized_action,q_anchor,statistics): ...
    def read_window(self,manifest_row:Mapping[str,Any],stage:str)->Mapping[str,Any]: ...

class LegacyOracle(Protocol):
    """Future R0 must preserve its own historical input/noise/statistics contract."""
    def predict_velocity(self,historical_inputs,noise,tau): ...

RESERVED_ADAPTERS={'piper':'e82.model.PiperWAM','tianji':None}
RESERVED_ORACLES={'P0_R0':None}
