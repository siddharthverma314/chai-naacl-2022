from .base import Transform
from .dict_transform import Flatten, Unflatten
from .one_hot_transform import OneHot, UnOneHot


class OneHotFlatten(Transform):
    def __init__(self, space):
        one_hot = OneHot(space)
        flatten = Flatten(one_hot.after_space)
        super().__init__(one_hot.before_space, flatten.after_space)
        self.one_hot = one_hot
        self.flatten = flatten

    def forward(self, x):
        return self.flatten(self.one_hot(x))


class UnOneHotUnflatten(Transform):
    def __init__(self, space):
        un_one_hot = UnOneHot(space)
        unflatten = Unflatten(un_one_hot.before_space)
        super().__init__(unflatten.before_space, un_one_hot.after_space)
        self.unflatten = unflatten
        self.un_one_hot = un_one_hot

    def forward(self, x):
        return self.un_one_hot(self.unflatten(x))
