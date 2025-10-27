import torch
import neps
from functools import partial
from typing import Literal, Callable


def scale_by_constant(constant):
    return partial(torch.mul, other=constant)


def clamp_by_constant(constant):
    return partial(torch.clamp, min=-constant, max=constant)


def resolve_expression(expr, var_dict):
    # recursively evaluate nested tuples of the form (callable, arg1, arg2, ...)
    if isinstance(expr, str):
        return var_dict.get(expr, expr)
    op, *args = expr
    # resolve nested args first (bottom-up)
    args = [resolve_expression(a, var_dict) for a in args]

    for n, arg in enumerate(args):
        if isinstance(arg, str):
            args[n] = var_dict[arg]
    return op(*args)


class NOSSpace(neps.PipelineSpace):
    def __init__(
        self,
        **_,
    ):

        self._variables = neps.Categorical(choices=("w", "g", "u", "v1", "v2"))

        self._constants = neps.Categorical(choices=(10, 1, 0, 0.1, 0.01, 0.9, 0.99))

        self._unary_funct = neps.Categorical(
            choices=(
                neps.Operation(
                    scale_by_constant,
                    kwargs={"constant": neps.Resampled(self._constants)},
                ),
                neps.Operation(
                    clamp_by_constant,
                    kwargs={"constant": neps.Resampled(self._constants)},
                ),
                torch.reciprocal,
                torch.square,
                torch.exp,
                torch.sqrt,
                torch.log,
                torch.neg,
            )
        )

        self._binary_funct = neps.Categorical(
            choices=(
                torch.add,
                torch.mul,
            )
        )

        self._element = neps.Categorical(
            choices=(
                neps.Resampled(self._variables),
                neps.Resampled(
                    neps.Operation(
                        operator=self.unaryFunction,
                        args=(
                            neps.Resampled(self._unary_funct),
                            neps.Resampled("_element"),
                        ),
                    )
                ),
                neps.Resampled(
                    neps.Operation(
                        operator=self.binaryFunction,
                        args=(
                            neps.Resampled(self._binary_funct),
                            neps.Resampled("_element"),
                            neps.Resampled("_element"),
                        ),
                    ),
                ),
            ),
            prior=0,
            prior_confidence="low",
        )

        self.optimizer_cls = neps.Operation(
            operator=self.create_optimizer,
            args=(
                neps.Resampled(self._element),
                neps.Resampled(self._element),
                neps.Resampled(self._element),
            ),
        )

    @staticmethod
    def unaryFunction(operation: Callable, input_value):
        return operation, input_value

    @staticmethod
    def binaryFunction(operation: Callable, input1, input2):
        return operation, input1, input2

    def create_optimizer(self, v1_line, v2_line, u_line):

        # Return a custom Optimizer class that captures the sampled expressions.
        class CustomOptimizer(torch.optim.Optimizer):
            def __init__(self, params, lr=1e-3, variables=(0.1, 0.1)):
                defaults = dict(lr=lr, vars=variables)
                super().__init__(params, defaults)
                # Ensure state entries for each parameter: v1, v2, u
                self.lr = lr
                for group in self.param_groups:
                    for p in group.get("params", []):
                        state = self.state.setdefault(p, {})
                        if "v1" not in state:
                            state["v1"] = torch.ones_like(p.data) * variables[0]
                        if "v2" not in state:
                            state["v2"] = torch.ones_like(p.data) * variables[1]
                        if "u" not in state:
                            state["u"] = torch.zeros_like(p.data)

            def step(self, closure=None):

                loss = None
                if closure is not None:
                    with torch.enable_grad():
                        loss = closure()

                for group in self.param_groups:
                    for p in group.get("params", []):
                        # get gradient (or zero if missing)
                        if p.grad is None:
                            d_p = torch.zeros_like(p.data)
                        else:
                            d_p = p.grad

                        state = self.state.setdefault(p, {})

                        # helper to coerce resolved values to tensors matching parameter
                        def as_tensor(x):
                            if isinstance(x, torch.Tensor):
                                # move/convert dtype to match parameter
                                try:
                                    return x.to(device=p.data.device, dtype=p.data.dtype)
                                except Exception:
                                    return x
                            else:
                                return torch.tensor(
                                    x, dtype=p.data.dtype, device=p.data.device
                                )

                        # build var dict with current values
                        var_dict = {
                            "g": d_p,
                            "w": p.data,
                            "u": state.get("u", torch.zeros_like(p.data)),
                            "v1": state.get("v1", torch.zeros_like(p.data)),
                            "v2": state.get("v2", torch.zeros_like(p.data)),
                        }

                        # evaluate and store new v1
                        new_v1 = resolve_expression(v1_line, var_dict)
                        new_v1 = as_tensor(new_v1)
                        state["v1"] = new_v1
                        var_dict["v1"] = new_v1

                        # evaluate and store new v2
                        new_v2 = resolve_expression(v2_line, var_dict)
                        new_v2 = as_tensor(new_v2)
                        state["v2"] = new_v2
                        var_dict["v2"] = new_v2

                        # evaluate update (u)
                        update = resolve_expression(u_line, var_dict)
                        update = as_tensor(update)
                        state["u"] = update

                        # apply update to parameter
                        p.data = p.data - self.lr * update

                return loss

        return CustomOptimizer


class NOSSpace2(neps.PipelineSpace):
    def __init__(
        self,
        max_lines=10,
        **_,
    ):

        self._variables = neps.Categorical(choices=("w", "g", "u", "v1", "v2"))

        self._constants = neps.Categorical(choices=(10, 1, 0, 0.1, 0.01, 0.9, 0.99))

        self._unary_funct = neps.Categorical(
            choices=(
                neps.Operation(
                    scale_by_constant,
                    kwargs={"constant": neps.Resampled(self._constants)},
                ),
                neps.Operation(
                    clamp_by_constant,
                    kwargs={"constant": neps.Resampled(self._constants)},
                ),
                torch.reciprocal,
                torch.square,
                torch.exp,
                torch.sqrt,
                torch.log,
                torch.neg,
            )
        )

        self._binary_funct = neps.Categorical(
            choices=(
                torch.add,
                torch.mul,
            )
        )

        self._line = neps.Categorical(
            choices=(
                (neps.Resampled(self._variables), neps.Resampled(self._variables)),
                (
                    neps.Resampled(self._variables),
                    neps.Resampled(
                        neps.Operation(
                            operator=self.unaryFunction,
                            args=(
                                neps.Resampled(self._unary_funct),
                                neps.Resampled(self._variables),
                            ),
                        )
                    ),
                ),
                (
                    neps.Resampled(self._variables),
                    neps.Resampled(
                        neps.Operation(
                            operator=self.binaryFunction,
                            args=(
                                neps.Resampled(self._binary_funct),
                                neps.Resampled(self._variables),
                                neps.Resampled(self._variables),
                            ),
                        )
                    ),
                ),
            ),
        )

        self._shared_lines = [neps.Resampled(self._line) for _ in range(max_lines)]

        self._line_choices = tuple(
            tuple(self._shared_lines[:i]) for i in range(1, max_lines + 1)
        )

        self.optimizer_cls = neps.Operation(
            operator=self.create_optimizer,
            args=(neps.Categorical(choices=self._line_choices)),
        )

        self.learning_rate = neps.Float(min_value=1e-5, max_value=1e-2, log=True)

    @staticmethod
    def unaryFunction(operation: Callable, input_value):
        return operation, input_value

    @staticmethod
    def binaryFunction(operation: Callable, input1, input2):
        return operation, input1, input2

    def create_optimizer(self, *lines):

        # Return a custom Optimizer class that captures the sampled expressions.
        class CustomOptimizer(torch.optim.Optimizer):
            def __init__(self, params, lr=1e-3, variables=(0.1, 0.1)):
                defaults = dict(lr=lr, vars=variables)
                super().__init__(params, defaults)
                # Ensure state entries for each parameter: v1, v2, u
                self.lr = lr
                for group in self.param_groups:
                    for p in group.get("params", []):
                        state = self.state.setdefault(p, {})
                        if "v1" not in state:
                            state["v1"] = torch.ones_like(p.data) * variables[0]
                        if "v2" not in state:
                            state["v2"] = torch.ones_like(p.data) * variables[1]
                        if "u" not in state:
                            state["u"] = torch.zeros_like(p.data)

            def step(self, closure=None):

                loss = None
                if closure is not None:
                    with torch.enable_grad():
                        loss = closure()

                for group in self.param_groups:
                    for p in group.get("params", []):
                        # get gradient (or zero if missing)
                        if p.grad is None:
                            d_p = torch.zeros_like(p.data)
                        else:
                            d_p = p.grad

                        state = self.state.setdefault(p, {})

                        # helper to coerce resolved values to tensors matching parameter
                        def as_tensor(x):
                            if isinstance(x, torch.Tensor):
                                # move/convert dtype to match parameter
                                try:
                                    return x.to(device=p.data.device, dtype=p.data.dtype)
                                except Exception:
                                    return x
                            else:
                                return torch.tensor(
                                    x, dtype=p.data.dtype, device=p.data.device
                                )

                        # build var dict with current values
                        var_dict = {
                            "g": d_p,
                            "w": p.data,
                            "u": state.get("u", torch.zeros_like(p.data)),
                            "v1": state.get("v1", torch.zeros_like(p.data)),
                            "v2": state.get("v2", torch.zeros_like(p.data)),
                        }

                        # evaluate and store new v1
                        for line in lines:
                            target_var, expr = line
                            result = resolve_expression(expr, var_dict)
                            result = as_tensor(result)
                            state[target_var] = result
                            var_dict[target_var] = result

                        # apply update to parameter
                        p.data = p.data - self.lr * state["u"]

                return loss

            def __repr__(self) -> str:
                string = f"{self.__class__.__name__}(\n"
                for group in self.param_groups:
                    string += f"  Parameter group:\n"
                    for k, v in group.items():
                        if k != "params":
                            string += f"    {k}: {v}\n"
                string += ")\nLines:\n"
                for line in lines:
                    string += f"  {line}\n"
                return string

        return CustomOptimizer
