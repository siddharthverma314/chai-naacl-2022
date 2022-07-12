from neural_chat.utils import create_random_space, torchify
from neural_chat.replay_buffer import ReplayBuffer
from flatten_dict import flatten
from gym.spaces import Discrete
import torch


def test_integration():
    for _ in range(100):
        for device in "cpu", "cuda":
            obs_space = create_random_space()
            act_space = create_random_space()
            buf = ReplayBuffer(obs_space, act_space, int(1e5), 1, device)
            print(buf.log_hyperparams())
            print("OBSSPEC", obs_space)
            print("ACTSPEC", act_space)

            step = {
                "obs": torchify(obs_space.sample(), device),
                "act": torchify(act_space.sample(), device),
                "rew": torchify(1.0, device),
                "next_obs": torchify(obs_space.sample(), device),
                "done": torchify(0, device),
            }
            buf.add(step)

            step2 = buf.sample()
            step = flatten(step)
            step2 = flatten(step2)
            assert step.keys() == step2.keys()
            for k in step:
                assert torch.all(step[k].cpu() == step2[k].cpu())

            print(buf.log_epoch())


def test_indices():
    BATCH_SIZE = 1000
    STEP_SIZE = 100

    space = Discrete(BATCH_SIZE)
    buf = ReplayBuffer(space, space, BATCH_SIZE, BATCH_SIZE, "cpu")
    r = torch.arange(BATCH_SIZE)
    for i in range(0, BATCH_SIZE, STEP_SIZE):
        mb = r[i : i + STEP_SIZE].unsqueeze(1)
        buf.add(
            {
                "obs": mb,
                "act": mb,
                "rew": mb,
                "next_obs": mb,
                "done": torch.zeros_like(mb),
            }
        )
    for _ in range(STEP_SIZE):
        sample = buf.sample(BATCH_SIZE, True)
        assert torch.all(sample["obs"] == sample["index"])
