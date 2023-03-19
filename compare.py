from src.DiscontinuousLayer import DiscontinuousLayer
from src.train_nn import create_data, train_nn_main
import torch
import matplotlib.pyplot as plt
from src.DiscontinuousLayer import heaviside

# training parameters
comparisons = [ {"maintenance": False, "discontinuous":True, "first_order_loss":True}, 
                {"maintenance": False, "discontinuous":False, "first_order_loss":False},
                ]
number_nns = 1
have_legend = True
colors = ["r", "b", "g", "orange", "purple"]


# data paramaters
low, high = 0, 1
number_data_points =8
maintence_points = 10
# function = lambda x: torch.sin(2 * torch.pi * x)
# function = lambda x: heaviside(x - 0.5)
# function = lambda x: torch.exp(x)
# function = lambda x: torch.sin(x) * torch.cos(x)
# function = lambda x: -(x-0.5)**3 + (x-0.5)**2
# function = lambda x: -64*(x-0.5)**4 + 16*(x-0.5)**2
# function = lambda x: (1-heaviside(x-0.5)) * (2*(x-0.5)**2 + 1)  + heaviside(x-0.5) * (-2*(x-0.5)**2 - 1)
function = lambda x: (1-heaviside(x-0.5)) * (-64*(x-0.5)**4 + 16*(x-0.5)**2-1)  + heaviside(x-0.5) * -(-64*(x-0.5)**4 + 16*(x-0.5)**2-1)

# function = lambda x: x
torch.manual_seed(1)

# generate training data
xs, ys, dys = create_data(function, number_data_points, low=low, high=high, uniform=True)
xs_maintenance, _, _ = create_data(function, maintence_points, low=low, high=high, uniform=True)

# create nns
trained_nns = []
for settings in comparisons:
    nns = [train_nn_main(xs, ys, dys=dys, maintenance_points=xs_maintenance, maximum_gradient_steps=100_000, minimum_loss=0.000001, activation_function=torch.nn.Sigmoid, **settings)[0] for i in range(number_nns)]
    trained_nns.append(nns)

# show output - plot true function and sampled points
xs_dense, ys_dense, _ = create_data(function, 100, low=low, high=high, uniform=True)
plt.plot(xs_dense.detach().cpu(), ys_dense.detach().cpu(), color='black', label="True function")
plt.scatter(xs.detach().cpu(), ys.detach().cpu(), color='black', label="Data points")

# plot nns
for color, nns, settings in zip(colors, trained_nns, comparisons):
    for i, nn in enumerate(nns):
        ys_dense = nn(xs_dense)
        label = "{} {} {} NN".format("Maintained" if settings["maintenance"] else "Unmaintained", 
                                    "Discontinuous" if settings["discontinuous"] else "Continuous", 
                                    "1st order" if settings["first_order_loss"] else "0th order", 
                                    ) if i == 0 else None
        plt.plot(xs_dense.detach().cpu(), ys_dense.detach().cpu(), color=color, alpha=0.5, label=label)



if have_legend: plt.legend()
plt.title("Comparing NN training methods under low data conditions (N={})".format(number_data_points))
plt.savefig("results/Compare.png")


# print(trained_nns[1][0][3].epsilon.data)
