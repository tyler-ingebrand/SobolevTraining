import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from tqdm import trange
from matplotlib.animation import FuncAnimation
from src.train_nn import create_data,  train_nn_main
from src.DiscontinuousLayer import heaviside

# data
comparisons = [ {"maintenance": False, "discontinuous":True, "first_order_loss":True},
                {"maintenance": False, "discontinuous":False, "first_order_loss":False},
                 ]
have_legend = True
colors = ["r", "b", "g", "orange", "purple"]
total_steps = 100_000
steps_per_frame = 25
new_data_every_n_steps = 100_000
torch.manual_seed(1)

# data paramaters
low, high = 0, 1
number_data_points = 8
maintence_points = 10

# function = lambda x: torch.sin(2 * torch.pi * x)
# function = lambda x: torch.sigmoid(100*(x-0.5))
# function = lambda x: heaviside(x - 0.5)
# function = lambda x: (1-heaviside(x-0.5)) * (2*(x-0.5)**2 + 1)  + heaviside(x-0.5) * (-2*(x-0.5)**2 - 1)
function = lambda x: (1-heaviside(x-0.5)) * (-64*(x-0.5)**4 + 16*(x-0.5)**2-1)  + heaviside(x-0.5) * -(-64*(x-0.5)**4 + 16*(x-0.5)**2-1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# generate points and graphs
xs_dense, ys_dense, dys_dense = create_data(function, 100, low=low, high=high, uniform=True)
y_low, y_high = min(ys_dense.cpu()), max(ys_dense.cpu())
y_low -= (y_high - y_low) * 0.05
y_high += (y_high - y_low) * 0.05
xs, ys, dys = None, None, None
xs_maintain, _, _ = create_data(function, maintence_points, low=low, high=high, uniform=True)


# generate NN variables and graphing data
nns, opts = [], []
for settings in comparisons:
    nn, opt = train_nn_main(xs_dense, 
                            ys_dense, 
                            **settings,
                            maximum_gradient_steps=0, minimum_loss=0.0,
                            dys=dys_dense, 
                            maintenance_points=xs_maintain,
                            model=None, 
                            opt=None, 
                            progress_bar=False)
    nns.append(nn)
    opts.append(opt)


# create animation
fig, ax = plt.subplots()
fig.set_size_inches(w=19, h=10, forward=True)
def animate(i):    # create plot area
    ax.clear()

    # plot true function
    lines = []

    l1, = ax.plot(xs_dense.detach().cpu(), function(xs_dense.detach().cpu()), color='black', label="True function")
    lines.append(l1)

    global xs, ys, dys
    if i % int(new_data_every_n_steps / steps_per_frame) == 0:
        xs, ys, dys = create_data(function, number_data_points, low=low, high=high, uniform=True)
        print("At frame {} out of {}".format(i, int(total_steps/steps_per_frame)))
    l2 = ax.scatter(xs.detach().cpu(), ys.detach().cpu(), color='black', label="Data points")
    lines.append(l2)

    # train nns
    global nns, opts
    for i, settings in enumerate(comparisons):
        nn, opt = train_nn_main(xs, 
                            ys, 
                            dys=dys,
                            **settings,
                            maximum_gradient_steps=steps_per_frame, 
                            minimum_loss=0.000001,
                            maintenance_points=xs_maintain,
                            model=nns[i], 
                            opt=opts[i], 
                            progress_bar=False)
        nns[i] = nn
        opts[i] = opt
        

        # show output - plot
        ys_dense = nn(xs_dense)
        label = "{} {} {} NN".format("Maintained" if settings["maintenance"] else "Unmaintained", 
                                    "Discontinuous" if settings["discontinuous"] else "Continuous", 
                                    "1st order" if settings["first_order_loss"] else "0th order", 
                                    )        
        line, = ax.plot(xs_dense.detach().cpu(), ys_dense.detach().cpu(), color=colors[i], label=label)
        lines.append(line)



    # add legend and title
    ax.legend(loc='upper right')
    ax.set_title("Comparing NN training methods under low data conditions (N={})".format(number_data_points))
    ax.set_xlim(low, high)
    ax.set_ylim(y_low, y_high)
    return lines


anim = FuncAnimation(fig, animate, frames=int(total_steps/steps_per_frame), interval=20, blit=True)
anim.save('results/movie123.mp4', dpi=100)
