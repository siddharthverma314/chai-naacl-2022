from neural_chat.utils import create_random_space, torchify, untorchify
from neural_chat.actor import DictActor


def test_integration():
    for _ in range(1000):
        obs_space = create_random_space()
        act_space = create_random_space()

        if not act_space.__class__ == "Dict":
            continue

        policy = DictActor(obs_space, act_space, [256, 256])

        obs = torchify(obs_space.sample())
        act, log_prob = policy.action_and_log_prob(obs)
        assert act_space.contains(untorchify(act))
