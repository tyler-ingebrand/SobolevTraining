from src.train_nn import train_nn, train_nn_first_order
import torch
import matplotlib.pyplot as plt

# training parameters
number_nns = 10

# data paramaters
low, high = 0, 1
number_data_points = 50
function = lambda x: torch.sin(4 * torch.pi * x)
# function = lambda x: torch.exp(x)
# function = lambda x: torch.sin(x) * torch.cos(x)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

# generate training data
xs = torch.linspace(low, high, number_data_points).reshape(-1, 1).to(device).requires_grad_(True)
ys = function(xs)
ys.backward(torch.ones_like(xs))
dys = torch.clone(xs.grad).detach()

# recreate without grads
xs = torch.linspace(low, high, number_data_points).reshape(-1, 1).to(device)
ys = function(xs)

# create nns
trained_nns =[
                    train_nn_first_order(xs, ys, dys,
                    maximum_gradient_steps=100_000, minimum_loss=0.000001, activation_function=torch.nn.Sigmoid)
              for i in range(number_nns)
            ]

# show output - plot
xs_dense = torch.linspace(low, high, 100).reshape(-1, 1).to(device)
for nn in trained_nns:
    ys_dense = nn(xs_dense)
    plt.plot(xs_dense.detach().cpu(), ys_dense.detach().cpu(), color='b', alpha=0.5)
plt.plot(xs_dense.detach().cpu(), function(xs_dense.detach().cpu()), color='black')
plt.scatter(xs.detach().cpu(), ys.detach().cpu(), color='b')
plt.show()

