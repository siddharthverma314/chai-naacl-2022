from torch.distributions.transforms import (
    TanhTransform,
    AffineTransform,
    ComposeTransform,
)


class TanhScaleTransform(ComposeTransform):
    def __init__(self, low, high):
        m = (high - low) / 2
        b = (high + low) / 2
        super().__init__([TanhTransform(), AffineTransform(b, m)])
