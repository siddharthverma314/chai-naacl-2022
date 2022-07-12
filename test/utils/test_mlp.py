from neural_chat.utils import MLP
import torch
from torch.functional import F


def test_simple_network():
    mlp = MLP(10, [60, 50], 5)
    t = torch.rand((50, 10))
    output = mlp.forward(t).detach()
    assert output.shape == (50, 5)

    output2 = mlp.cuda().forward(t.cuda()).detach().cpu()
    assert output2.shape == (50, 5)


def test_logging_integration():
    mlp = MLP(10, [60, 50], 5)
    print(mlp.log_hyperparams())
    print(mlp.log_epoch())


def test_fake_task():
    X = torch.rand((5000, 100)) * 100
    A = torch.rand((100, 10))
    y = X @ A

    X = X.cuda()
    y = y.cuda()
    mlp = MLP(100, [], 10).cuda()
    opt = torch.optim.Adam(mlp.parameters())

    for _ in range(10000):
        loss = F.mse_loss(y, mlp.forward(X))
        opt.zero_grad()
        loss.backward()
        opt.step()

    assert F.mse_loss(mlp.state_dict()["mlp.0.weight"].T.cpu(), A).item() < 1e-2
