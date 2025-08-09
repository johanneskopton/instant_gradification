from autograd import V, ElSum, ReLU, Exp
import numpy as np
import torch


def test_element_wise():
    a = V(np.array([3, -35, 3]))
    b = V(np.array([4, 4, 4]))
    c = V(np.array([2.5, 2, 4]))
    d = ElSum(ReLU(a + Exp(b) / c))

    d.eval()
    d.back()

    aa = torch.tensor([3, -35, 3], dtype=torch.float, requires_grad=True)
    bb = torch.tensor([4, 4, 4], dtype=torch.float, requires_grad=True)
    cc = torch.tensor([2.5, 2, 4], dtype=torch.float, requires_grad=True)
    dd = torch.nn.ReLU()(aa + torch.exp(bb) / cc).sum()

    dd.backward()
    assert np.isclose(d.value, dd.detach().numpy()).all()
    assert np.isclose(a.grad, aa.grad.numpy()).all()
    assert np.isclose(b.grad, bb.grad.numpy()).all()
    assert np.isclose(c.grad, cc.grad.numpy()).all()


test_element_wise()
