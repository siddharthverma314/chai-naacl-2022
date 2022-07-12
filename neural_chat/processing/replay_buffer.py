from neural_chat.replay_buffer import ReplayBuffer
from neural_chat.utils import torchify
import torch
import json
import gym


def populate_replay_buffer_from_craigslist(
    env: gym.Env, datafile: str, device: str, rb: ReplayBuffer
) -> None:
    with open(datafile, "r") as f:
        data = json.load(f)

    data = [
        env.offline_process(d["state"], d["action"], d["reward"], d["done"])
        for d in data
    ]
    data = [[torchify(x, rb.device) for x in d] for d in data]
    for (d1, d2) in zip(data, data[1:]):
        d1o, d1a, d1r, d1d = d1
        d2o, d2a, d2r, d2d = d2
        d = {"obs": d1o, "act": d1a, "rew": d1r, "done": d1d, "next_obs": d2o}
        rb.add(d)


def populate_response_map_from_craigslist(
    env: gym.Env, response_file: str, device: torch.device
) -> torch.Tensor:
    with open(response_file, "r") as f:
        responses = json.load(f)

    actions = []

    fake_obs = {
        "utterance": "bleh",
        "offer": 0.5,
        "action_type": 0,
        "context": "nothing",
    }
    fake_act = lambda utterance: {
        "utterance": utterance,
        "offer": 0.0,
        "action_type": 0,
    }
    for action_lst in responses:
        local_actions = []
        for action in action_lst:
            _, action, _, _ = env.offline_process(fake_obs, fake_act(action), 0, False)
            local_actions.append(torch.Tensor(action["utterance"]))
        actions.append(torch.stack(local_actions))
    actions = torch.stack(actions)

    return actions.to(device)


# (realgud:pdb "rm -rvf /tmp/log; python fixed_lm.py --logdir /tmp/log --dataset=../data/out.json --response-map=/nfs/kun2/users/vsiddharth/research/neural_chat/gpt2-finetune-try4/database_cleaned.json")
