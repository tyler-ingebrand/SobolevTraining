from src.train_nn import train_nn
import torch
import matplotlib.pyplot as plt

# training parameters
number_nns = 10

# data paramaters
low, high = 0, 1
number_data_points = 3
function = lambda x: torch.sin(2 * torch.pi * x)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# generate training data
xs = torch.linspace(low, high, number_data_points, requires_grad=False).reshape(-1, 1).to(device)
ys = function(xs).to(device)
#ys.backward()
#dys = xs.grad
#print(dys)
#err

# create nns
trained_nns =[train_nn(xs, ys, maximum_gradient_steps=100_000, minimum_loss=0.000001, activation_function=torch.nn.Sigmoid) for i in range(number_nns)]

# show output - plot
xs_dense = torch.linspace(low, high, 100).reshape(-1, 1).to(device)
for nn in trained_nns:
    ys_dense = nn(xs_dense)
    plt.plot(xs_dense.detach().cpu(), ys_dense.detach().cpu(), color='b', alpha=0.5)
plt.plot(xs_dense.detach().cpu(), function(xs_dense.detach().cpu()), color='black')
plt.scatter(xs.detach().cpu(), ys.detach().cpu(), color='b')
plt.show()

