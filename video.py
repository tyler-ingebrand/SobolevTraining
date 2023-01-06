import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from tqdm import trange
from matplotlib.animation import FuncAnimation
from src.train_nn import create_data, train_nn, train_nn_first_order

# data
total_steps = 100_000
steps_per_frame = 25
new_data_every_n_steps = 1_000
torch.manual_seed(1)

# data paramaters
low, high = 0, 1
number_data_points = 3
function = lambda x: torch.sin(2 * torch.pi * x)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# generate NN variables and graphing data
trained_nn,trained_nn2,opt,opt2 = None, None, None, None
xs_dense, ys_dense, dys = create_data(function, 100, low=low, high=high, uniform=True)
y_low, y_high = min(ys_dense.cpu()), max(ys_dense.cpu())
y_low -= (y_high - y_low) * 0.05
y_high += (y_high - y_low) * 0.05
xs, ys, dys = None, None, None

# create animation
fig, ax = plt.subplots()
fig.set_size_inches(w=10, h=10, forward=True)
def animate(i):    # create plot area
    ax.clear()

    # plot true function
    l1, = ax.plot(xs_dense.detach().cpu(), function(xs_dense.detach().cpu()), color='black', label="True function")


    global xs, ys, dys
    if i % int(new_data_every_n_steps / steps_per_frame) == 0:
        xs, ys, dys = create_data(function, number_data_points, low=low, high=high, uniform=False)
        print("At frame {} out of {}".format(i, int(total_steps/steps_per_frame)))
    l4 = ax.scatter(xs.detach().cpu(), ys.detach().cpu(), color='black', label="Data points")

    # train nns
    global trained_nn, trained_nn2, opt, opt2
    trained_nn, opt = train_nn(xs, ys, maximum_gradient_steps=steps_per_frame, minimum_loss=0.000001,
                               activation_function=torch.nn.Sigmoid, model= trained_nn, opt=opt, progress_bar=False)
    trained_nn2, opt2 = train_nn_first_order(xs, ys, dys, maximum_gradient_steps=steps_per_frame, minimum_loss=0.000001,
                                             activation_function=torch.nn.Sigmoid, model=trained_nn2, opt=opt2, progress_bar=False)
    # show output - plot
    ys_dense = trained_nn(xs_dense)
    label = "0th order approximation"
    l2, = ax.plot(xs_dense.detach().cpu(), ys_dense.detach().cpu(), color='r', label=label)

    ys_dense2 = trained_nn2(xs_dense)
    label = "0th + 1st order approximation (Sobolev)"
    l3, = ax.plot(xs_dense.detach().cpu(), ys_dense2.detach().cpu(), color='b', label=label)

    # add legend and title
    ax.legend(loc='upper right')
    ax.set_title("Comparing NN training methods under low data conditions (N={})".format(number_data_points))
    ax.set_xlim(low, high)
    ax.set_ylim(y_low, y_high)
    return l1, l2, l3, l4


anim = FuncAnimation(fig, animate, frames=int(total_steps/steps_per_frame), interval=20, blit=True)
anim.save('movie.mp4', dpi=100)
