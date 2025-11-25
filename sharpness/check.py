import torch
import torch.nn as nn
from torch.autograd.functional import vhp, hvp
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.func import functional_call

model = nn.Linear(5, 1)
criterion = nn.MSELoss()

x = torch.randn(3, 5)
target = torch.randn(3, 1)

y = model(x)
loss = criterion(y, target)

heta = parameters_to_vector(model.parameters())
functional_call(model, theta, x)

print("loss:", loss.item())

# def func(model:nn.Module, theta: torch.Tensor) -> torch.Tensor:
#     outputs = functional_call(model, theta, x)
#     vector_to_parameters(theta, model.parameters())
#     outputs = model(x)
#     loss = criterion(outputs, target)
#     return loss

# # Hessian-vector product: works with tensors, not parameters
# theta = parameters_to_vector(model.parameters())
# v = torch.randn_like(theta)
# print("flat_params.device", theta.device)
# print("v.device", v.device)

# hvp(func, theta, v)

from torch.func import functional_call

params = dict(model.named_parameters())
buffers = dict(model.named_buffers())

def func(params, buffers, x, target):
    y = functional_call(model, (params, buffers), (x,))
    return torch.nn.MSELoss()(y, target)


# # Utilities to make nn.Module functional
# def del_attr(obj, names):
#     if len(names) == 1:
#         delattr(obj, names[0])
#     else:
#         del_attr(getattr(obj, names[0]), names[1:])
# def set_attr(obj, names, val):
#     if len(names) == 1:
#         setattr(obj, names[0], val)
#     else:
#         set_attr(getattr(obj, names[0]), names[1:], val)

# def make_functional(mod):
#     orig_params = tuple(mod.parameters())
#     # Remove all the parameters in the model
#     names = []
#     for name, p in list(mod.named_parameters()):
#         del_attr(mod, name.split("."))
#         names.append(name)
#     return orig_params, names

# def load_weights(mod, names, params):
#     for name, p in zip(names, params):
#         set_attr(mod, name.split("."), p)

# N = 10
# model = models.resnet18(pretrained=False)
# criterion = torch.nn.CrossEntropyLoss()

# params, names = make_functional(model)
# # Make params regular Tensors instead of nn.Parameter
# params = tuple(p.detach().requires_grad_() for p in params)

# inputs = torch.rand([N, 3, 224, 224], device=device)
# labels = torch.rand(N, device=device).mul(10).long()
# def f(*new_params):
#     load_weights(model, names, new_params)
#     out = model(inputs)

#     loss = criterion(out, labels)
#     return loss

# vhp(f, params, grads)