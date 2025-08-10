from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from nn import NN, MSE
from autograd import C
import numpy as np

plt.style.use("bmh")


# Load dataset
X_set, y_set = make_moons(noise=0.2, random_state=0)

# Create Neural Net
nn = NN([2, 10, 10, 1])

# Train
LR = 0.1
LR_FAC = 0.998
EPOCHS = 1000

mses = np.zeros(EPOCHS)
lr = LR
for j in range(EPOCHS):
    errors = np.zeros(len(X_set))
    order = np.random.permutation(np.arange(len(X_set)))
    for i in range(len(X_set)):
        X = X_set[order[i]]
        y = nn.forward(C(X))
        y_hat = y_set[order[i]]
        y.eval()
        error = y.value - y_hat
        y.back(error * LR)
        y.reset_grad()
        errors[i] = error
    mse = MSE(errors)
    mses[j] = mse
    lr *= LR_FAC
    if j % 10 == 0:
        print(f"EPOCH: {j}\t\t MSE: {mse}\t\t LR: {lr}")

## Visualize Results

x_res = 80
y_res = 50
x_grid = np.linspace(np.min(X_set, axis=0)[0], np.max(X_set, axis=0)[0], x_res)
y_grid = np.linspace(np.min(X_set, axis=0)[1], np.max(X_set, axis=0)[1], y_res)

xv, yv = np.meshgrid(x_grid, y_grid)
grid = np.vstack([xv.flatten(), yv.flatten()]).T
grid_values = np.zeros(len(grid))

for i, X in enumerate(grid):
    y = nn.forward(C(X))
    y.eval()
    grid_values[i] = y.value[0]

fig, ax = plt.subplots()

ax.scatter(grid[:, 0], grid[:, 1], c=grid_values, marker="s", s=30, alpha=1)
ax.scatter(X_set[:, 0], X_set[:, 1], c=y_set, cmap="magma")
ax.set_title("classification results")

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1)
ax.plot(mses)
ax.set_xlabel("epoch")
ax.set_ylabel("mse")

plt.show()
