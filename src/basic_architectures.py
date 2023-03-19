import torch
from typing import List
from src.DiscontinuousLayer import DiscontinuousLayer

def get_model(model_description, device=None):
    if model_description["type"] == "mlp":
        nn = get_mlp(**model_description)
    elif model_description["type"] == "d-mlp":
        nn = get_discontinuous_mlp(**model_description)
    else:
        raise Exception("Model not specified correctly")
    device = torch.device(device if device is not None else "cuda:0" if torch.cuda.is_available() else "cpu")
    return nn.to(device)


def get_mlp(n_inputs, n_outputs, hiddens:List[int]=[64,64], activation=torch.nn.ReLU, **kw_args):
    hiddens.insert(0, n_inputs)
    hiddens.append(n_outputs)
    nn_layers = []
    for i, (first, second) in enumerate(zip(hiddens[:-1], hiddens[1:])):
        nn_layers.append(torch.nn.Linear(first, second))
        if i != len(hiddens) - 2:
            nn_layers.append(activation())
    nn = torch.nn.Sequential(*nn_layers)
    return nn

def get_discontinuous_mlp(n_inputs, n_outputs, hiddens:List[int]=[64,64], discontinuous_positions:List[int]=[1], activation=torch.nn.ReLU, **kw_args):
    hiddens.insert(0, n_inputs)
    hiddens.append(n_outputs)
    nn_layers = []
    for i, (first, second) in enumerate(zip(hiddens[:-1], hiddens[1:])):
        nn_layers.append(torch.nn.Linear(first, second))
        if i < len(hiddens) - 2:
            if i in discontinuous_positions:
                nn_layers.append(DiscontinuousLayer(second, activation_function=activation()))
            else:
                nn_layers.append(activation())
    nn = torch.nn.Sequential(*nn_layers)
    return nn