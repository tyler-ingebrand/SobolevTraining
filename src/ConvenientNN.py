from typing import Union, Callable, List
import torch
from tqdm import trange

from src.DiscontinuousLayer import DiscontinuousLayer
from src.basic_architectures import *

def get_gradient_norm(params, order=2):
    all_norms = [torch.norm(p.grad.detach(), order) if p.grad is not None else torch.tensor(0.0, device=p.device) for p in params]
    # total_norm = torch.norm(torch.stack(all_norms), order)
    return torch.stack(all_norms)


class ConvenientNN:
    def __init__(self,
                 model:Union[torch.nn.Module, dict], # should be a torch NN module, or a dict specifying how to construct it
                 loss_functions:Union[Callable, List[Callable], dict[Callable, float]], # Should be a loss function or a list/dict of loss functions. Should be of the form (model, kw_args) -> loss scalar
                 optimizer_class=torch.optim.Adam, # The optimizer alg to use
                 optimizer_kw_args={}, # the args to use to init the optimizer
                 use_first_order_weights=False, # whether or not to scale all gradient norms to the same value or not
                 first_order_norm_minimum=0.0001, # the minimum accepted norm for a gradient. Grads below this are considered converged.
                 first_order_norm_base_index = 0, # the loss function for which all other norms should be scaled to, if using first order weighting
                 early_terminate_loss=float('-inf'), # if loss is less than this value, training terminates early
                 ):

        # verify inputs make sense
        assert isinstance(loss_functions, Callable) or len(loss_functions) >= 1, "At least 1 loss function must be provided. Got {} loss functions".format(len(loss_functions))
        assert first_order_norm_base_index < len(loss_functions), f"First order norm index must be less than the number of loss functions, got {first_order_norm_base_index}, expected a value less than {len(loss_functions)}"

        # create model based on model dict if not given, save it
        if isinstance(model, dict):
            model = get_model(model)
        self.model = model

        # save loss functions
        if isinstance(loss_functions, Callable):
            loss_functions = {loss_functions : 1.0}
        if isinstance(loss_functions, List):
            loss_functions = {lf : 1.0 for lf in loss_functions}
        self.loss_functions = loss_functions

        # generate optimzer based on provided class and KW arguments
        self.opt = optimizer_class(model.parameters(), **optimizer_kw_args)

        # save training variables
        self.use_first_order_weights = use_first_order_weights
        self.first_order_norm_minimum = first_order_norm_minimum
        self.early_terminate_loss = early_terminate_loss
        self.first_order_norm_base_index = first_order_norm_base_index

        # init variables
        self.adjusted_weights = torch.ones(len(loss_functions))
        self.total_training_steps = 0
        self.adjusted_history = []
   
   
    def forward(self, xs, deterministic=False):
        if deterministic:
            self.model.eval()
        ret = self.model(xs)
        if deterministic:
            self.model.train()
        return ret

    def __call__(self, xs, deterministic=False):
        return self.forward(xs, deterministic)

    def train(self,
              descent_steps:int=1,
              progress_bar=False,
              **loss_function_kwargs
              ):

        assert descent_steps >= 1
        r = trange(descent_steps) if progress_bar else range(descent_steps)
        for step in r:
            # Compute norms if we scale by norms every 100 steps
            if self.use_first_order_weights and self.total_training_steps % 100 == 0:
                self._update_norms(loss_function_kwargs)
                #print(self.adjusted_weights.clone().detach())
                #err
                # self.adjusted_history.append(self.adjusted_weights.clone().detach())
                # print(self.adjusted_weights)

            # compute sum of all losses
            total_loss = 0.0
            unnormed_loss = 0.0
            for (loss_fcn, weight), adjusted_weight in zip(self.loss_functions.items(), self.adjusted_weights):
                loss = loss_fcn(self.model, **loss_function_kwargs)
                total_loss += loss * weight * adjusted_weight

                if self.use_first_order_weights and self.total_training_steps % 100 == 0:       
                    l = self.adjusted_weights.clone().detach()
                    l = torch.cat((l, total_loss.clone().detach().cpu().view(-1)))
                    self.adjusted_history.append(l)

            # backpropagate
            self.opt.zero_grad()
            total_loss.backward()
            self.opt.step()

            self.total_training_steps += 1
            if total_loss < self.early_terminate_loss:
                break
        return total_loss

    def _update_norms(self, loss_function_kwargs):
        # get all norms for each parameter for each loss function
        norms = []
        for i, loss_fcn in enumerate(self.loss_functions.keys()):
            norms.append(self._get_norms_for_loss_function(loss_fcn, loss_function_kwargs))
        
        # get baseline norm to normalize to. Its still a list of norms for each parameter
        baseline_norms = norms[self.first_order_norm_base_index]

        # for all loss functions, compute a fair baseline 
        # We only consider parameters of the baseline for parameters whose norm is 0 in the loss function
        for i, loss_fcn in enumerate(self.loss_functions.keys()):
            this_loss_fcn_norms = norms[i]
            this_norm = torch.norm(this_loss_fcn_norms)
            fair_baseline_norm = torch.norm(torch.where(this_loss_fcn_norms != 0.0, baseline_norms, 0.0))
            alpha = 0.3
            new_weight = (1-alpha) * self.adjusted_weights[i] + alpha * max(fair_baseline_norm, self.first_order_norm_minimum) / max(this_norm, self.first_order_norm_minimum) 
            self.adjusted_weights[i] = new_weight
            # self.adjusted_weights[i] = max(fair_baseline_norm, self.first_order_norm_minimum) / max(this_norm, self.first_order_norm_minimum) 
        

    def _get_norms_for_loss_function(self, loss_function, loss_function_kwargs):
        self.opt.zero_grad()
        loss = loss_function(self.model, **loss_function_kwargs)
        loss.backward()
        all_param_norms = get_gradient_norm(self.model.parameters())
        self.opt.zero_grad()
        return all_param_norms