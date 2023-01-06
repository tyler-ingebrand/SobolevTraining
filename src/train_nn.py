import torch
import torch.nn as nn
from tqdm import trange

# inputs expcted to be batchsize x input size,
# outputs expected to be batchsize x outputsize
def train_nn(inputs:torch.tensor, outputs:torch.tensor,
             maximum_gradient_steps, minimum_loss, activation_function=nn.ReLU,
             model=None, opt=None, progress_bar=True):

    assert len(inputs.shape) == 2 and len(outputs.shape) == 2, "Inputs and outputs must have 2 dimensions each"
    assert inputs.shape[0] == outputs.shape[0], "Input batch size must match output batch size"

    # fetch sizes from data
    batch_size = inputs.shape[0]
    input_size = inputs.shape[1]
    output_size = outputs.shape[1]

    # create NN
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = nn.Sequential(nn.Linear(input_size, 64), activation_function(),
                          nn.Linear(64, 64), activation_function(),
                          nn.Linear(64, output_size)).to(device)
    if opt is None:
        opt = torch.optim.Adam(model.parameters())
    loss_function = nn.MSELoss()

    # optimize
    r = trange(maximum_gradient_steps) if progress_bar else range(maximum_gradient_steps)
    for i in r:
        opt.zero_grad()
        loss = loss_function(model(inputs), outputs)
        loss.backward()
        opt.step()
        if loss < minimum_loss:
            break

    return model, opt

# inputs expcted to be batchsize x input size,
# outputs expected to be batchsize x outputsize
def train_nn_first_order(inputs:torch.tensor, outputs:torch.tensor, gradients:torch.tensor,
                        maximum_gradient_steps, minimum_loss, activation_function=nn.ReLU,
                         model=None, opt=None, progress_bar=True):

    assert len(inputs.shape) == 2 and len(outputs.shape) == 2 and len(gradients.shape) == 2, "Inputs and outputs must have 2 dimensions each"
    assert inputs.shape[0] == outputs.shape[0] == gradients.shape[0], "Input batch size must match output batch size"

    # fetch sizes from data
    batch_size = inputs.shape[0]
    input_size = inputs.shape[1]
    output_size = outputs.shape[1]

    # create NN
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = nn.Sequential(nn.Linear(input_size, 64), activation_function(),
                          nn.Linear(64, 64), activation_function(),
                          nn.Linear(64, output_size)).to(device)
    if opt is None:
        opt = torch.optim.Adam(model.parameters())
    loss_function1 = nn.MSELoss()
    loss_function2 = nn.MSELoss()

    # optimize
    r = trange(maximum_gradient_steps) if progress_bar else range(maximum_gradient_steps)
    for i in r:
        # compute gradient based loss function
        #inputs.grad = None
        #inputs.requires_grad_(True)
        y_hats = model(inputs)
        dy_hats = torch.autograd.grad(y_hats, inputs, torch.ones_like(y_hats), create_graph=True)[0] # [0] because we only have 1 input and it returns a tuple

        opt.zero_grad() # remove current gradients we used to calculate dy_hats. We will compute gradients for loss next

        # calculate gradient loss
        loss2 = loss_function2(dy_hats, gradients)
        loss2.backward()

        # compute normal loss function
        loss1 = loss_function1(model(inputs), outputs)
        loss1.backward()

        # step optimizer
        opt.step()

        if loss1 +  loss2 < minimum_loss:
             break

    return model, opt

def create_data(function, number_points, low=0.0, high=1.0, uniform=True, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Generate xs
    if uniform:
        xs = torch.linspace(low, high, number_points).reshape(-1, 1).to(device).requires_grad_(True)
    else:
        xs = ((high - low) * torch.rand(number_points, 1) + low).to(device).requires_grad_(True)

    # compute ys and dys/dxs
    ys = function(xs)
    dys = torch.autograd.grad(ys, xs, torch.ones_like(ys))
    ys = ys.detach()
    dys = dys[0].detach()
    return xs, ys, dys