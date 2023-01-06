from src.train_nn import train_nn, train_nn_first_order, create_data
import torch
import matplotlib.pyplot as plt

# training parameters
have_legend = True
number_training_graphs = 10
torch.manual_seed(1)

# data paramaters
low, high = 0, 1
number_data_points = 6
function = lambda x: torch.sin(2 * torch.pi * x)
# function = lambda x: torch.exp(x)
# function = lambda x: torch.sin(x) * torch.cos(x)
# function = lambda x: -(x-0.5)**3 + (x-0.5)**2
# function = lambda x: -64*(x-0.5)**4 + 16*(x-0.5)**2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trained_nn,trained_nn2,opt,opt2 = None, None, None, None
xs_dense, ys_dense, dys = create_data(function, 100, low=low, high=high, uniform=True)
plt.plot(xs_dense.detach().cpu(), function(xs_dense.detach().cpu()), color='black', label="True function")

for i in range(number_training_graphs):
    # generate training data
    xs, ys, dys = create_data(function, number_data_points, low=low, high=high, uniform=False)

    # create nns
    trained_nn, opt = train_nn(xs, ys, maximum_gradient_steps=100_000, minimum_loss=0.000001, activation_function=torch.nn.Sigmoid, model=trained_nn, opt=opt)
    trained_nn2, opt2 = train_nn_first_order(xs, ys, dys, maximum_gradient_steps=100_000, minimum_loss=0.000001, activation_function=torch.nn.Sigmoid, model=trained_nn2, opt=opt2)

    # show output - plot
    ys_dense = trained_nn(xs_dense)
    label = "0th order approximation" if i == 0 else None
    plt.plot(xs_dense.detach().cpu(), ys_dense.detach().cpu(), color='r', alpha=0.1 + i*0.1, label=label)

    ys_dense2 = trained_nn2(xs_dense)
    label = "0th + 1st order approximation (Sobolev)" if i == 0 else None
    plt.plot(xs_dense.detach().cpu(), ys_dense2.detach().cpu(), color='b', alpha=0.1 + i*0.1, label=label)

if have_legend: plt.legend()
plt.title("Comparing NN training methods under low data conditions (N={})".format(number_data_points))
plt.show()

