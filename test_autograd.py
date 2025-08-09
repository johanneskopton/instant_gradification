from autograd import V, ElSum, ReLU, Exp, sigmoid
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


def test_nn():
    aa = np.array(
        [[6, -4, 0, 2, -8], [-8, 8, -4, 3, 5], [-2, 1, 4, -5, 7], [1, 0, -7, 4, 5]]
    )
    aa2 = np.array([[2, -2, 0, 2], [7, 0, -3, -5], [-7, 9, 2, -2], [-1, -9, 2, 3]])
    bb = np.array([1, 6, 2, 3])
    bb2 = np.array([1, 3, 2, -3])

    inputr = np.array([1, 2, -3, -4, 2])

    A = V(aa)
    A2 = V(aa2)
    b = V(bb)
    b2 = V(bb2)

    input = V(inputr)
    c = ReLU(A @ input + b)
    c2 = sigmoid(A2 @ c + b2)

    c2.eval()

    c2.back()

    AA = torch.tensor(aa, dtype=float, requires_grad=True)
    AA2 = torch.tensor(aa2, dtype=float, requires_grad=True)
    BB = torch.tensor(bb, dtype=float, requires_grad=True)
    BB2 = torch.tensor(bb2, dtype=float, requires_grad=True)
    INPUT = torch.tensor(inputr, dtype=float, requires_grad=True)
    CC = torch.nn.ReLU()(AA @ INPUT + BB)
    CC2 = torch.nn.Sigmoid()(AA2 @ CC + BB2)

    CC2.sum().backward()

    assert np.isclose(c2.value, CC2.detach().numpy()).all()
    assert np.isclose(A.grad, AA.grad.numpy()).all()
    assert np.isclose(A2.grad, AA2.grad.numpy()).all()
    assert np.isclose(b.grad, BB.grad.numpy()).all()
    assert np.isclose(b2.grad, BB2.grad.numpy()).all()


test_element_wise()
test_nn()
