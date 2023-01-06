from src.train_nn import train_nn, train_nn_first_order, create_data
import torch
import matplotlib.pyplot as plt

# training parameters
number_nns = 10
have_legend = True

# data paramaters
low, high = 0, 1
number_data_points = 3
function = lambda x: torch.sin(2 * torch.pi * x)
# function = lambda x: torch.exp(x)
# function = lambda x: torch.sin(x) * torch.cos(x)
# function = lambda x: -(x-0.5)**3 + (x-0.5)**2
# function = lambda x: -64*(x-0.5)**4 + 16*(x-0.5)**2
torch.manual_seed(1)

# generate training data
xs, ys, dys = create_data(function, number_data_points, low=low, high=high, uniform=True)

# create nns
trained_nns =[train_nn(xs, ys, maximum_gradient_steps=100_000, minimum_loss=0.000001, activation_function=torch.nn.Sigmoid)[0] for i in range(number_nns)]
trained_nns2 =[train_nn_first_order(xs, ys, dys, maximum_gradient_steps=100_000, minimum_loss=0.000001, activation_function=torch.nn.Sigmoid)[0] for i in range(number_nns)]

# show output - plot
xs_dense, ys_dense, _ = create_data(function, 100, low=low, high=high, uniform=True)
plt.plot(xs_dense.detach().cpu(), ys_dense.detach().cpu(), color='black', label="True function")
plt.scatter(xs.detach().cpu(), ys.detach().cpu(), color='black', label="Data points")

for i, nn in enumerate(trained_nns):
    ys_dense = nn(xs_dense)
    label = "0th order approximation" if i == 0 else None
    plt.plot(xs_dense.detach().cpu(), ys_dense.detach().cpu(), color='r', alpha=0.5, label=label)
for i, nn in enumerate(trained_nns2):
    ys_dense = nn(xs_dense)
    label = "0th + 1st order approximation (Sobolev)" if i == 0 else None
    plt.plot(xs_dense.detach().cpu(), ys_dense.detach().cpu(), color='b', alpha=0.5, label=label)

if have_legend: plt.legend()
plt.title("Comparing NN training methods under low data conditions (N={})".format(number_data_points))
plt.show()

