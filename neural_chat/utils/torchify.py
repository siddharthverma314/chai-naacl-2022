from typing import Union
import torch
import numpy as np


def torchify(
    obs: Union[np.ndarray, dict], device: Union[str, torch.device] = "cpu"
) -> Union[torch.Tensor, dict]:
    if isinstance(obs, dict):
        ret = {}
        for k, v in obs.items():
            tv = torchify(v, device)
            if tv is not None:
                ret[k] = tv
        return ret
    obs = torch.tensor(obs)
    if obs.dtype != torch.int64 or obs.dtype != torch.int32:
        obs = obs.float()
    while obs.dim() < 2:
        obs = obs.unsqueeze(0)
    return obs.to(device)


def untorchify(obs: Union[torch.Tensor, dict]) -> Union[np.ndarray, dict]:
    if isinstance(obs, dict):
        return {k: untorchify(v) for k, v in obs.items()}
    obs = obs.detach().cpu()
    if torch.is_floating_point(obs):
        obs = obs.squeeze(0)
    else:
        obs = obs.squeeze()
    return obs.numpy()
