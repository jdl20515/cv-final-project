from .baseline import ESPCN
from .temporal_loss import TemporalESPCN
from .multiframe import MultiFrameSR
from .recurrent import RecurrentSR
from .flow_warp import FlowWarpSR

MODEL_REGISTRY = {
    "baseline": ESPCN,
    "temporal_loss": TemporalESPCN,
    "multiframe": MultiFrameSR,
    "recurrent": RecurrentSR,
    "flow_warp": FlowWarpSR,
}


def build_model(config):
    name = config["model"]["name"]
    params = config["model"].get("params", {})
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**params)
