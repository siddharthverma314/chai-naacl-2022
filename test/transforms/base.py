from neural_chat.utils import collate, create_random_space, torchify
import torch


NUM_SAMPLES = 100


def custom_equals(a, b):
    if isinstance(a, dict):
        assert isinstance(b, dict) and b.keys() == a.keys()
        for k, v in a.items():
            custom_equals(v, b[k])
    else:
        print("ISCLOSE:", torch.isclose(a.float(), b.float()))
        assert (
            isinstance(b, torch.Tensor)
            and b.dim() == 2
            and torch.all(torch.isclose(a.float(), b.float()))
        )


def make_test_multi_space(model, size=20):
    def test_multi_space():
        for _ in range(NUM_SAMPLES):
            space = create_random_space()
            m = model(space)

            samples = []
            for _ in range(size):
                samples.append({"obs": space.sample()})
            sample = collate([torchify(s) for s in samples])["obs"]

            print("=" * 50)
            print("SPACE:", space)
            print("SAMPLE:", sample)
            forward = m.forward(sample)
            print("FORWARD:", forward)
            custom_equals(forward, sample)

    return test_multi_space


def make_test_single_space(model):
    return make_test_multi_space(model, 1)
