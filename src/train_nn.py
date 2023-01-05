import torch
import torch.nn as nn
from tqdm import trange

# inputs expcted to be batchsize x input size,
# outputs expected to be batchsize x outputsize
def train_nn(inputs:torch.tensor, outputs:torch.tensor,
             maximum_gradient_steps, minimum_loss, activation_function=nn.ReLU,
             model=None):

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
    optimizer = torch.optim.Adam(model.parameters())
    loss_function = nn.MSELoss()

    # optimize
    for i in trange(maximum_gradient_steps):
        optimizer.zero_grad()
        loss = loss_function(model(inputs), outputs)
        loss.backward()
        optimizer.step()
        if loss < minimum_loss:
            break

    return model

# inputs expcted to be batchsize x input size,
# outputs expected to be batchsize x outputsize
def train_nn_first_order(inputs:torch.tensor, outputs:torch.tensor, gradients:torch.tensor,
                        maximum_gradient_steps, minimum_loss, activation_function=nn.ReLU,
                         model=None):

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
    optimizer = torch.optim.Adam(model.parameters())
    loss_function1 = nn.MSELoss()
    loss_function2 = nn.MSELoss()

    # optimize
    for i in trange(maximum_gradient_steps):
        # compute gradient based loss function
        inputs.grad = None
        inputs.requires_grad_(True)
        y_hats = model(inputs)
        y_hats.backward(torch.ones_like(inputs), create_graph=True)
        dy_hats = inputs.grad

        optimizer.zero_grad() # remove current gradients we used to calculate dy_hats. We will compute gradients for loss next

        # calculate gradient loss
        loss2 = loss_function2(dy_hats, gradients)
        loss2.backward()

        # compute normal loss function
        loss1 = loss_function1(model(inputs), outputs)
        loss1.backward()

        # step optimizer
        optimizer.step()

        # clean up so we dont create memory leak with torch
        for p in model.parameters():
            p.grad = None

        if loss1 +  loss2 < minimum_loss:
             break

    return model
