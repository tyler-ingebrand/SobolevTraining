import torch
import torch.nn as nn
from tqdm import trange

from src.DiscontinuousLayer import DiscontinuousLayer
import matplotlib.pyplot as plt

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


# inputs expcted to be batchsize x input size,
# outputs expected to be batchsize x outputsize
def train_nn_discontinuous(inputs:torch.tensor, outputs:torch.tensor,
             maximum_gradient_steps, minimum_loss, activation_function=nn.ReLU,
             progress_bar=True, nns=None, opts=None):

    assert len(inputs.shape) == 2 and len(outputs.shape) == 2, "Inputs and outputs must have 2 dimensions each"
    assert inputs.shape[0] == outputs.shape[0], "Input batch size must match output batch size"

    # fetch sizes from data
    batch_size = inputs.shape[0]
    input_size = inputs.shape[1]
    output_size = outputs.shape[1]

    # create NN
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if nns is None:
        approx1 = nn.Sequential(nn.Linear(input_size, 64), activation_function(),
                                      nn.Linear(64, 64), activation_function(),
                                      nn.Linear(64, output_size)).to(device)
        approx2 = nn.Sequential(nn.Linear(input_size, 64), activation_function(),
                                    nn.Linear(64, 64), activation_function(),
                                    nn.Linear(64, output_size)).to(device)
        classifier = nn.Sequential(nn.Linear(input_size, 64), activation_function(),
                                        nn.Linear(64, 64), activation_function(),
                                        nn.Linear(64, 2),
                                        ).to(device)
    else:
        approx1 = nns[0]
        approx2 = nns[1]
        classifier = nns[2]

    if opts is None:
        opt_approx = torch.optim.Adam((*approx1.parameters(), *approx2.parameters()), lr=0.001)
        opt_classifier = torch.optim.Adam(classifier.parameters(), lr=0.0001)
    else:
        opt_approx = opts[0]
        opt_classifier = opts[1]


    def model(x):
        y1 = approx1(x)
        y2 = approx2(x)
        logits = classifier(x)
        probs = torch.nn.Softmax(dim=1)(logits)

        # Non continuous method
        # max_values, max_inidcies = torch.max(probs, dim=1)
        # y1[max_inidcies==1] = y2[max_inidcies==1]
        # return y1

        # smooth method
        ret = y1 * probs[:, 0].reshape(-1, 1) + y2 * probs[:, 1].reshape(-1, 1)
        return ret

    # optimize
    r = trange(maximum_gradient_steps) if progress_bar else range(maximum_gradient_steps)
    for i in r:
        with torch.no_grad():
            # Update classifier
            # Need to use CrossEntropy
            out1 = approx1(inputs)
            out2 = approx2(inputs)
            outs = torch.cat((out1, out2), dim=1)
            distances = torch.abs(outs - outputs)
            min_values, min_indicies = torch.min(distances, dim=1)
        loss1 = torch.nn.CrossEntropyLoss()(classifier(inputs), min_indicies) # + torch.nn.MSELoss()(model(inputs), outputs)
        opt_classifier.zero_grad()
        loss1.backward()
        opt_classifier.step()
        # print(classifier[0])

        # update others
        with torch.no_grad():
            logits = classifier(inputs)
            probs = torch.nn.Softmax(dim=1)(logits)
            weights = probs ** 10
            # predictions = model(inputs)

        loss2 = torch.tensor([0.0], device=device)
        opt_approx.zero_grad()

        for b in range(batch_size):
                if min_indicies[b] == 0:
                    loss2 += (approx1(inputs[b, :]) - outputs[b, :]) ** 2 * weights[b, 0]
                else:
                    loss2 += (approx2(inputs[b, :]) - outputs[b, :]) ** 2 * weights[b, 1]


        loss2.backward()
        opt_approx.step()

        # early termination
        if loss1 + loss2 < minimum_loss:
            break


    return model, (opt_approx, opt_classifier), (approx1, approx2, classifier)



def get_gradient_norm(params, order=2):
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), order) for p in params]), order)
    return total_norm

# inputs expcted to be batchsize x input size,
# outputs expected to be batchsize x outputsize
def train_nn_main(      xs:torch.tensor, 
                        ys:torch.tensor, 
                        discontinuous:bool,  # whether or not to create a discontinuous NN
                        first_order_loss:bool, # whether or not to use a first order loss function
                        maintenance:bool,   # whether or not to use maintence points
                        maximum_gradient_steps, minimum_loss,
                        dys:torch.tensor=None, 
                        maintenance_points:torch.tensor=None,
                        activation_function=nn.ReLU,
                        model=None, 
                        opt=None, 
                        progress_bar=True):
    assert not maintenance or  xs.shape[1] == maintenance_points.shape[1], "Inputs and maintence points must have the same size, got {} {}".format(xs.shape,maintenance_points.shape)
    assert len(xs.shape) == 2 and len(ys.shape) == 2, "Inputs and outputs must have 2 dimensions each"
    assert xs.shape[0] == ys.shape[0], "Input batch size must match output batch size"
    assert first_order_loss == False or dys is not None, "If using first order loss, must provide the derivatives"

    if progress_bar:
        print("Settings:")
        print("\tDiscontinuous:", discontinuous)
        print("\tFirst order loss:", first_order_loss)
        print("\tMaintenance:", maintenance)


    # fetch sizes from data
    batch_size = xs.shape[0]
    input_size = xs.shape[1]
    output_size = ys.shape[1]

    # create NN and opt if not exists
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model is None:
        if discontinuous:
            model = nn.Sequential(  nn.Linear(input_size, 64), activation_function(),
                                    nn.Linear(64, 64), DiscontinuousLayer(64, activation_function=activation_function()),
                                    nn.Linear(64, output_size)).to(device)

        else:
            model = nn.Sequential(nn.Linear(input_size, 64), activation_function(),
                            nn.Linear(64, 64), activation_function(),
                            nn.Linear(64, output_size)).to(device)
    if opt is None:
        opt1 = torch.optim.Adam(model.parameters()) # , lr=0.01)
        if first_order_loss:
            opt2 = torch.optim.Adam(model.parameters())
    else:
        if isinstance(opt, tuple):
            opt1 = opt[0]
            opt2 = opt[1]
        else:
            opt1 = opt

    # optimize
    r = trange(maximum_gradient_steps) if progress_bar else range(maximum_gradient_steps)
    for i in r:
        
        # compute maintence gradients if using it
        if maintenance:
            
            # calculate how much a change in every gradient impacts the output
            # For each maintenance point, compute gradient of output wrt parameters
            # Take absolute value, since we care about a change in either direction
            # take average across all points.
            d_y_maintenance_wrt_params = [torch.zeros_like(p) for p in model.parameters()]
            for point in maintenance_points:
                    opt1.zero_grad()
                    y_maintain = model(point)
                    dy_maintain = torch.autograd.grad(y_maintain, model.parameters(), torch.ones_like(y_maintain))
                    for index in range(len(dy_maintain)):
                        d_y_maintenance_wrt_params[index] += torch.abs(dy_maintain[index])
            

            # # normalize for each parameter and
            d_flattened = torch.concat([d_p.view(-1) for d_p in d_y_maintenance_wrt_params])
            max_gradient = max(torch.max(flattened) for flattened in d_y_maintenance_wrt_params) * 1.01
            #mean = torch.mean(d_flattened)
            #std = torch.std(d_flattened)
            # # plot g
            # # grads_flattened = torch.abs(torch.concat([p.view(-1) for p in model.parameters()]))
            # # plt.scatter(grads_flattened.detach().cpu(), d_flattened.detach().cpu())
            # # plt.xlabel("Loss function gradients")
            # # plt.ylabel("Output function gradients")
            # # plt.savefig("CompareGrads.png")
            # # plt.clf()
            # # err1
            # # end plot



        # 0th order loss
        opt1.zero_grad()
        loss = nn.MSELoss()(model(xs), ys)
        loss.backward()
        opt1.step()

        # 1st order loss
        th = 0.01
        if first_order_loss:
            zeroth_order_norm = get_gradient_norm(model.parameters())

            # compute gradient based loss function
            y_hats = model(xs)
            dy_hats, = torch.autograd.grad(y_hats, xs, torch.ones_like(y_hats), create_graph=True)
            opt2.zero_grad()  # remove current gradients we used to calculate dy_hats. We will compute gradients for loss next

            # calculate gradient loss
            loss_first_order = nn.MSELoss()(dy_hats, dys)
            loss_first_order.backward()

            # scale to same size as 0th order loss
            first_order_norm = get_gradient_norm(model.parameters())
            for i, p in enumerate(model.parameters()):
                p.grad *= max(th, zeroth_order_norm) / max(th, first_order_norm)
            opt2.step()

        if maintenance:
            # before step, need to modify derivatives
            for p, d_p in zip(model.parameters(), d_y_maintenance_wrt_params):
                # normed = (d_p - mean)/std
                # inverted = -normed
                # shifted = inverted + 1
                # cut_negatives = torch.where(shifted >= 0.0, shifted, 0.0)
                # p.grad *= cut_negatives
                importance_weights = 1 - d_p/max_gradient
                p.grad *= importance_weights

        loss_break = loss + (loss_first_order if first_order_loss else 0)
        if loss_break < minimum_loss:
            break
    return model, opt1 if not first_order_loss else (opt1, opt2)