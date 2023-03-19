from src.ConvenientNN import ConvenientNN
from src.DiscontinuousLayer import DiscontinuousLayer
from src.train_nn import create_data, train_nn_main
import torch
import matplotlib.pyplot as plt
from src.DiscontinuousLayer import heaviside
import numpy as np


if __name__ == "__main__":
    low, high = 0, 1
    number_data_points = 6
    function = lambda x: (1-heaviside(x-0.5)) * (-64*(x-0.5)**4 + 16*(x-0.5)**2-1)  + heaviside(x-0.5) * -(-64*(x-0.5)**4 + 16*(x-0.5)**2-1)
    torch.manual_seed(1)
    xs, ys, dys = create_data(function, number_data_points, low=low, high=high, uniform=True)


    def first_order_loss(m, xs, dys, **kw_args):
        y_hats = m(xs)
        dy_hats, = torch.autograd.grad(y_hats, xs, torch.ones_like(y_hats), create_graph=True)
        for p in m.parameters():
            p.grad = None
        return torch.nn.MSELoss()(dy_hats, dys)

    # model_desc = {"type":"mlp", "n_inputs":1, "n_outputs":1}
    model_desc = {"type":"d-mlp", "n_inputs":1, "n_outputs":1 }
    model = ConvenientNN(model=model_desc,
                         loss_functions={first_order_loss : 0.1,
                                         lambda m,xs, ys, **kw_args: torch.nn.MSELoss()(m(xs), ys) : 1.0,
                                         lambda m, **kw_args: torch.linalg.norm(m[3].epsilon ,ord=1) : 0.01,
                                         },
                         use_first_order_weights=True,
                         first_order_norm_base_index= 1,
                         early_terminate_loss=0.0001
                         )
    loss = model.train(xs=xs, ys=ys, dys=dys, descent_steps=10_000, progress_bar=True, )

    # show output - plot true function and sampled points
    xs_dense, ys_dense, _ = create_data(function, 100, low=low, high=high, uniform=True)
    plt.plot(xs_dense.detach().cpu(), ys_dense.detach().cpu(), color='black', label="True function")
    plt.scatter(xs.detach().cpu(), ys.detach().cpu(), color='black', label="Data points")
    plt.plot(xs_dense.detach().cpu(), model.forward(xs_dense).detach().cpu(),label="NN")
    plt.legend()
    plt.title("Comparing NN training methods under low data conditions (N={})".format(number_data_points))
    plt.savefig("results/test.png")
    plt.clf()

    # show adjusted weight history
    d = torch.stack(model.adjusted_history)
    for i in range(d.shape[1]):
        d[:, i] /= torch.max(d[:, i]) 
    plt.plot(d)
    plt.legend(labels=["w1", "w2", "w3", "fair loss"])
    plt.savefig("results/history.png")

