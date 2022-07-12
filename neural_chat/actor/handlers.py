from typing import List
import abc
import torch.distributions as pyd
import torch
from gym.spaces import Box, Discrete, MultiDiscrete, Space
from neural_chat.transforms import TanhScaleTransform


class Handler(abc.ABC):
    @abc.abstractproperty
    def input_size(self) -> int:
        raise NotImplementedError

    @abc.abstractproperty
    def output_size(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, inp: torch.Tensor, deterministic=False) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def log_prob(self, inp: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BoxHandler(Handler):
    def __init__(self, box: Box):
        self.box = box

    @property
    def input_size(self):
        return 2 * len(self.box.low)

    @property
    def output_size(self):
        return len(self.box.low)

    def _get_dist(self, inp):
        mu, log_std = inp.chunk(2, 1)
        log_std = TanhScaleTransform(-5, 1)(log_std)
        dist = pyd.Normal(mu, torch.exp(log_std))
        return dist

    def sample(self, inp, deterministic=False):
        if deterministic:
            mu, _ = inp.chunk(2, 1)
            return mu
        return self._get_dist(inp).rsample()

    def log_prob(self, inp, sample):
        return self._get_dist(inp).log_prob(sample).sum(1, keepdim=True)


class DiscreteHandler(Handler):
    def __init__(self, discrete: Discrete):
        self.discrete = discrete

    @property
    def input_size(self):
        return int(self.discrete.n)

    @property
    def output_size(self):
        return 1

    def _get_dist(self, inp):
        return pyd.Categorical(logits=inp)

    def sample(self, inp, deterministic=False):
        if deterministic:
            return torch.argmax(inp, dim=1, keepdim=True)
        return self._get_dist(inp).sample().unsqueeze(1)

    def log_prob(self, inp, sample):
        return self._get_dist(inp).log_prob(sample.squeeze()).unsqueeze(1)


class MultiHandler(Handler):
    def __init__(self, handlers: List[Handler]):
        self.handlers = handlers

    @property
    def input_size(self):
        return sum([h.input_size for h in self.handlers])

    @property
    def output_size(self):
        return sum([h.output_size for h in self.handlers])

    def _split_input(self, inp):
        return inp.split([h.input_size for h in self.handlers], dim=1)

    def _split_output(self, out):
        return out.split([h.output_size for h in self.handlers], dim=1)

    def sample(self, inp, deterministic=False):
        return torch.cat(
            [
                h.sample(i, deterministic)
                for h, i in zip(self.handlers, self._split_input(inp))
            ],
            dim=1,
        )

    def log_prob(self, inp, sample):
        lprobs = [
            h.log_prob(i, s)
            for h, i, s in zip(
                self.handlers, self._split_input(inp), self._split_output(sample)
            )
        ]
        for handler, lprob in zip(self.handlers, lprobs):
            if torch.any(torch.isnan(lprob)):
                raise ValueError("NaN value found for handler: %s" % str(handler))
        return sum(lprobs)


class MultiDiscreteHandler(MultiHandler):
    def __init__(self, multidiscrete: MultiDiscrete):
        super().__init__([DiscreteHandler(Discrete(n)) for n in multidiscrete.nvec])
