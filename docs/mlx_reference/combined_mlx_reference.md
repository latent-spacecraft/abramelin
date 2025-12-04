# mlx_nn_layers

Source: https://ml-explore.github.io/mlx/build/html/python/nn.html

---

* [.rst](../_sources/python/nn.rst)
* .pdf

# Neural Networks

## Contents

# Neural Networks[#](#neural-networks "Link to this heading")

Writing arbitrarily complex neural networks in MLX can be done using only
[`mlx.core.array`](_autosummary/mlx.core.array.html#mlx.core.array "mlx.core.array") and [`mlx.core.value_and_grad()`](_autosummary/mlx.core.value_and_grad.html#mlx.core.value_and_grad "mlx.core.value_and_grad"). However, this requires the
user to write again and again the same simple neural network operations as well
as handle all the parameter state and initialization manually and explicitly.

The module `mlx.nn` solves this problem by providing an intuitive way of
composing neural network layers, initializing their parameters, freezing them
for finetuning and more.

## Quick Start with Neural Networks[#](#quick-start-with-neural-networks "Link to this heading")

```python
import mlx.core as mx
import mlx.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dims: int, out_dims: int):
        super().__init__()

        self.layers = [
            nn.Linear(in_dims, 128),
            nn.Linear(128, 128),
            nn.Linear(128, out_dims),
        ]

    def __call__(self, x):
        for i, l in enumerate(self.layers):
            x = mx.maximum(x, 0) if i > 0 else x
            x = l(x)
        return x

# The model is created with all its parameters but nothing is initialized
# yet because MLX is lazily evaluated
mlp = MLP(2, 10)

# We can access its parameters by calling mlp.parameters()
params = mlp.parameters()
print(params["layers"][0]["weight"].shape)

# Printing a parameter will cause it to be evaluated and thus initialized
print(params["layers"][0])

# We can also force evaluate all parameters to initialize the model
mx.eval(mlp.parameters())

# A simple loss function.
# NOTE: It doesn't matter how it uses the mlp model. It currently captures
#       it from the local scope. It could be a positional argument or a
#       keyword argument.
def l2_loss(x, y):
    y_hat = mlp(x)
    return (y_hat - y).square().mean()

# Calling `nn.value_and_grad` instead of `mx.value_and_grad` returns the
# gradient with respect to `mlp.trainable_parameters()`
loss_and_grad = nn.value_and_grad(mlp, l2_loss)
```

Copy to clipboard

## The Module Class[#](#the-module-class "Link to this heading")

The workhorse of any neural network library is the [`Module`](nn/module.html#mlx.nn.Module "mlx.nn.Module") class. In
MLX the [`Module`](nn/module.html#mlx.nn.Module "mlx.nn.Module") class is a container of [`mlx.core.array`](_autosummary/mlx.core.array.html#mlx.core.array "mlx.core.array") or
[`Module`](nn/module.html#mlx.nn.Module "mlx.nn.Module") instances. Its main function is to provide a way to
recursively **access** and **update** its parameters and those of its
submodules.

### Parameters[#](#parameters "Link to this heading")

A parameter of a module is any public member of type [`mlx.core.array`](_autosummary/mlx.core.array.html#mlx.core.array "mlx.core.array") (its
name should not start with `_`). It can be arbitrarily nested in other
[`Module`](nn/module.html#mlx.nn.Module "mlx.nn.Module") instances or lists and dictionaries.

[`Module.parameters()`](nn/_autosummary/mlx.nn.Module.parameters.html#mlx.nn.Module.parameters "mlx.nn.Module.parameters") can be used to extract a nested dictionary with all
the parameters of a module and its submodules.

A [`Module`](nn/module.html#mlx.nn.Module "mlx.nn.Module") can also keep track of “frozen” parameters. See the
[`Module.freeze()`](nn/_autosummary/mlx.nn.Module.freeze.html#mlx.nn.Module.freeze "mlx.nn.Module.freeze") method for more details. [`mlx.nn.value_and_grad()`](_autosummary/mlx.nn.value_and_grad.html#mlx.nn.value_and_grad "mlx.nn.value_and_grad")
the gradients returned will be with respect to these trainable parameters.

### Updating the Parameters[#](#updating-the-parameters "Link to this heading")

MLX modules allow accessing and updating individual parameters. However, most
times we need to update large subsets of a module’s parameters. This action is
performed by [`Module.update()`](nn/_autosummary/mlx.nn.Module.update.html#mlx.nn.Module.update "mlx.nn.Module.update").

### Inspecting Modules[#](#inspecting-modules "Link to this heading")

The simplest way to see the model architecture is to print it. Following along with
the above example, you can print the `MLP` with:

```python
print(mlp)
```

Copy to clipboard

This will display:

```python
MLP(
  (layers.0): Linear(input_dims=2, output_dims=128, bias=True)
  (layers.1): Linear(input_dims=128, output_dims=128, bias=True)
  (layers.2): Linear(input_dims=128, output_dims=10, bias=True)
)
```

Copy to clipboard

To get more detailed information on the arrays in a [`Module`](nn/module.html#mlx.nn.Module "mlx.nn.Module") you can use
[`mlx.utils.tree_map()`](_autosummary/mlx.utils.tree_map.html#mlx.utils.tree_map "mlx.utils.tree_map") on the parameters. For example, to see the shapes of
all the parameters in a [`Module`](nn/module.html#mlx.nn.Module "mlx.nn.Module") do:

```python
from mlx.utils import tree_map
shapes = tree_map(lambda p: p.shape, mlp.parameters())
```

Copy to clipboard

As another example, you can count the number of parameters in a [`Module`](nn/module.html#mlx.nn.Module "mlx.nn.Module")
with:

```python
from mlx.utils import tree_flatten
num_params = sum(v.size for _, v in tree_flatten(mlp.parameters()))
```

Copy to clipboard

## Value and Grad[#](#value-and-grad "Link to this heading")

Using a [`Module`](nn/module.html#mlx.nn.Module "mlx.nn.Module") does not preclude using MLX’s high order function
transformations ([`mlx.core.value_and_grad()`](_autosummary/mlx.core.value_and_grad.html#mlx.core.value_and_grad "mlx.core.value_and_grad"), [`mlx.core.grad()`](_autosummary/mlx.core.grad.html#mlx.core.grad "mlx.core.grad"), etc.). However,
these function transformations assume pure functions, namely the parameters
should be passed as an argument to the function being transformed.

There is an easy pattern to achieve that with MLX modules

```python
model = ...

def f(params, other_inputs):
    model.update(params)  # <---- Necessary to make the model use the passed parameters
    return model(other_inputs)

f(model.trainable_parameters(), mx.zeros((10,)))
```

Copy to clipboard

However, [`mlx.nn.value_and_grad()`](_autosummary/mlx.nn.value_and_grad.html#mlx.nn.value_and_grad "mlx.nn.value_and_grad") provides precisely this pattern and only
computes the gradients with respect to the trainable parameters of the model.

In detail:

* it wraps the passed function with a function that calls [`Module.update()`](nn/_autosummary/mlx.nn.Module.update.html#mlx.nn.Module.update "mlx.nn.Module.update")
  to make sure the model is using the provided parameters.
* it calls [`mlx.core.value_and_grad()`](_autosummary/mlx.core.value_and_grad.html#mlx.core.value_and_grad "mlx.core.value_and_grad") to transform the function into a function
  that also computes the gradients with respect to the passed parameters.
* it wraps the returned function with a function that passes the trainable
  parameters as the first argument to the function returned by
  [`mlx.core.value_and_grad()`](_autosummary/mlx.core.value_and_grad.html#mlx.core.value_and_grad "mlx.core.value_and_grad")

|  |  |
| --- | --- |
| [`value_and_grad`](_autosummary/mlx.nn.value_and_grad.html#mlx.nn.value_and_grad "mlx.nn.value_and_grad")(model, fn) | Transform the passed function `fn` to a function that computes the gradients of `fn` wrt the model's trainable parameters and also its value. |
| [`quantize`](_autosummary/mlx.nn.quantize.html#mlx.nn.quantize "mlx.nn.quantize")(model[, group\_size, bits, mode, ...]) | Quantize the sub-modules of a module according to a predicate. |
| [`average_gradients`](_autosummary/mlx.nn.average_gradients.html#mlx.nn.average_gradients "mlx.nn.average_gradients")(gradients[, group, ...]) | Average the gradients across the distributed processes in the passed group. |

* [Module](nn/module.html)
  + [`Module`](nn/module.html#mlx.nn.Module)
  + [mlx.nn.Module.training](nn/_autosummary/mlx.nn.Module.training.html)
    - [`Module.training`](nn/_autosummary/mlx.nn.Module.training.html#mlx.nn.Module.training)
  + [mlx.nn.Module.state](nn/_autosummary/mlx.nn.Module.state.html)
    - [`Module.state`](nn/_autosummary/mlx.nn.Module.state.html#mlx.nn.Module.state)
  + [mlx.nn.Module.apply](nn/_autosummary/mlx.nn.Module.apply.html)
    - [`Module.apply()`](nn/_autosummary/mlx.nn.Module.apply.html#mlx.nn.Module.apply)
  + [mlx.nn.Module.apply\_to\_modules](nn/_autosummary/mlx.nn.Module.apply_to_modules.html)
    - [`Module.apply_to_modules()`](nn/_autosummary/mlx.nn.Module.apply_to_modules.html#mlx.nn.Module.apply_to_modules)
  + [mlx.nn.Module.children](nn/_autosummary/mlx.nn.Module.children.html)
    - [`Module.children()`](nn/_autosummary/mlx.nn.Module.children.html#mlx.nn.Module.children)
  + [mlx.nn.Module.eval](nn/_autosummary/mlx.nn.Module.eval.html)
    - [`Module.eval()`](nn/_autosummary/mlx.nn.Module.eval.html#mlx.nn.Module.eval)
  + [mlx.nn.Module.filter\_and\_map](nn/_autosummary/mlx.nn.Module.filter_and_map.html)
    - [`Module.filter_and_map()`](nn/_autosummary/mlx.nn.Module.filter_and_map.html#mlx.nn.Module.filter_and_map)
  + [mlx.nn.Module.freeze](nn/_autosummary/mlx.nn.Module.freeze.html)
    - [`Module.freeze()`](nn/_autosummary/mlx.nn.Module.freeze.html#mlx.nn.Module.freeze)
  + [mlx.nn.Module.leaf\_modules](nn/_autosummary/mlx.nn.Module.leaf_modules.html)
    - [`Module.leaf_modules()`](nn/_autosummary/mlx.nn.Module.leaf_modules.html#mlx.nn.Module.leaf_modules)
  + [mlx.nn.Module.load\_weights](nn/_autosummary/mlx.nn.Module.load_weights.html)
    - [`Module.load_weights()`](nn/_autosummary/mlx.nn.Module.load_weights.html#mlx.nn.Module.load_weights)
  + [mlx.nn.Module.modules](nn/_autosummary/mlx.nn.Module.modules.html)
    - [`Module.modules()`](nn/_autosummary/mlx.nn.Module.modules.html#mlx.nn.Module.modules)
  + [mlx.nn.Module.named\_modules](nn/_autosummary/mlx.nn.Module.named_modules.html)
    - [`Module.named_modules()`](nn/_autosummary/mlx.nn.Module.named_modules.html#mlx.nn.Module.named_modules)
  + [mlx.nn.Module.parameters](nn/_autosummary/mlx.nn.Module.parameters.html)
    - [`Module.parameters()`](nn/_autosummary/mlx.nn.Module.parameters.html#mlx.nn.Module.parameters)
  + [mlx.nn.Module.save\_weights](nn/_autosummary/mlx.nn.Module.save_weights.html)
    - [`Module.save_weights()`](nn/_autosummary/mlx.nn.Module.save_weights.html#mlx.nn.Module.save_weights)
  + [mlx.nn.Module.set\_dtype](nn/_autosummary/mlx.nn.Module.set_dtype.html)
    - [`Module.set_dtype()`](nn/_autosummary/mlx.nn.Module.set_dtype.html#mlx.nn.Module.set_dtype)
  + [mlx.nn.Module.train](nn/_autosummary/mlx.nn.Module.train.html)
    - [`Module.train()`](nn/_autosummary/mlx.nn.Module.train.html#mlx.nn.Module.train)
  + [mlx.nn.Module.trainable\_parameters](nn/_autosummary/mlx.nn.Module.trainable_parameters.html)
    - [`Module.trainable_parameters()`](nn/_autosummary/mlx.nn.Module.trainable_parameters.html#mlx.nn.Module.trainable_parameters)
  + [mlx.nn.Module.unfreeze](nn/_autosummary/mlx.nn.Module.unfreeze.html)
    - [`Module.unfreeze()`](nn/_autosummary/mlx.nn.Module.unfreeze.html#mlx.nn.Module.unfreeze)
  + [mlx.nn.Module.update](nn/_autosummary/mlx.nn.Module.update.html)
    - [`Module.update()`](nn/_autosummary/mlx.nn.Module.update.html#mlx.nn.Module.update)
  + [mlx.nn.Module.update\_modules](nn/_autosummary/mlx.nn.Module.update_modules.html)
    - [`Module.update_modules()`](nn/_autosummary/mlx.nn.Module.update_modules.html#mlx.nn.Module.update_modules)
* [Layers](nn/layers.html)
  + [mlx.nn.ALiBi](nn/_autosummary/mlx.nn.ALiBi.html)
    - [`ALiBi`](nn/_autosummary/mlx.nn.ALiBi.html#mlx.nn.ALiBi)
  + [mlx.nn.AvgPool1d](nn/_autosummary/mlx.nn.AvgPool1d.html)
    - [`AvgPool1d`](nn/_autosummary/mlx.nn.AvgPool1d.html#mlx.nn.AvgPool1d)
  + [mlx.nn.AvgPool2d](nn/_autosummary/mlx.nn.AvgPool2d.html)
    - [`AvgPool2d`](nn/_autosummary/mlx.nn.AvgPool2d.html#mlx.nn.AvgPool2d)
  + [mlx.nn.AvgPool3d](nn/_autosummary/mlx.nn.AvgPool3d.html)
    - [`AvgPool3d`](nn/_autosummary/mlx.nn.AvgPool3d.html#mlx.nn.AvgPool3d)
  + [mlx.nn.BatchNorm](nn/_autosummary/mlx.nn.BatchNorm.html)
    - [`BatchNorm`](nn/_autosummary/mlx.nn.BatchNorm.html#mlx.nn.BatchNorm)
  + [mlx.nn.CELU](nn/_autosummary/mlx.nn.CELU.html)
    - [`CELU`](nn/_autosummary/mlx.nn.CELU.html#mlx.nn.CELU)
  + [mlx.nn.Conv1d](nn/_autosummary/mlx.nn.Conv1d.html)
    - [`Conv1d`](nn/_autosummary/mlx.nn.Conv1d.html#mlx.nn.Conv1d)
  + [mlx.nn.Conv2d](nn/_autosummary/mlx.nn.Conv2d.html)
    - [`Conv2d`](nn/_autosummary/mlx.nn.Conv2d.html#mlx.nn.Conv2d)
  + [mlx.nn.Conv3d](nn/_autosummary/mlx.nn.Conv3d.html)
    - [`Conv3d`](nn/_autosummary/mlx.nn.Conv3d.html#mlx.nn.Conv3d)
  + [mlx.nn.ConvTranspose1d](nn/_autosummary/mlx.nn.ConvTranspose1d.html)
    - [`ConvTranspose1d`](nn/_autosummary/mlx.nn.ConvTranspose1d.html#mlx.nn.ConvTranspose1d)
  + [mlx.nn.ConvTranspose2d](nn/_autosummary/mlx.nn.ConvTranspose2d.html)
    - [`ConvTranspose2d`](nn/_autosummary/mlx.nn.ConvTranspose2d.html#mlx.nn.ConvTranspose2d)
  + [mlx.nn.ConvTranspose3d](nn/_autosummary/mlx.nn.ConvTranspose3d.html)
    - [`ConvTranspose3d`](nn/_autosummary/mlx.nn.ConvTranspose3d.html#mlx.nn.ConvTranspose3d)
  + [mlx.nn.Dropout](nn/_autosummary/mlx.nn.Dropout.html)
    - [`Dropout`](nn/_autosummary/mlx.nn.Dropout.html#mlx.nn.Dropout)
  + [mlx.nn.Dropout2d](nn/_autosummary/mlx.nn.Dropout2d.html)
    - [`Dropout2d`](nn/_autosummary/mlx.nn.Dropout2d.html#mlx.nn.Dropout2d)
  + [mlx.nn.Dropout3d](nn/_autosummary/mlx.nn.Dropout3d.html)
    - [`Dropout3d`](nn/_autosummary/mlx.nn.Dropout3d.html#mlx.nn.Dropout3d)
  + [mlx.nn.Embedding](nn/_autosummary/mlx.nn.Embedding.html)
    - [`Embedding`](nn/_autosummary/mlx.nn.Embedding.html#mlx.nn.Embedding)
  + [mlx.nn.ELU](nn/_autosummary/mlx.nn.ELU.html)
    - [`ELU`](nn/_autosummary/mlx.nn.ELU.html#mlx.nn.ELU)
  + [mlx.nn.GELU](nn/_autosummary/mlx.nn.GELU.html)
    - [`GELU`](nn/_autosummary/mlx.nn.GELU.html#mlx.nn.GELU)
  + [mlx.nn.GLU](nn/_autosummary/mlx.nn.GLU.html)
    - [`GLU`](nn/_autosummary/mlx.nn.GLU.html#mlx.nn.GLU)
  + [mlx.nn.GroupNorm](nn/_autosummary/mlx.nn.GroupNorm.html)
    - [`GroupNorm`](nn/_autosummary/mlx.nn.GroupNorm.html#mlx.nn.GroupNorm)
  + [mlx.nn.GRU](nn/_autosummary/mlx.nn.GRU.html)
    - [`GRU`](nn/_autosummary/mlx.nn.GRU.html#mlx.nn.GRU)
  + [mlx.nn.HardShrink](nn/_autosummary/mlx.nn.HardShrink.html)
    - [`HardShrink`](nn/_autosummary/mlx.nn.HardShrink.html#mlx.nn.HardShrink)
  + [mlx.nn.HardTanh](nn/_autosummary/mlx.nn.HardTanh.html)
    - [`HardTanh`](nn/_autosummary/mlx.nn.HardTanh.html#mlx.nn.HardTanh)
  + [mlx.nn.Hardswish](nn/_autosummary/mlx.nn.Hardswish.html)
    - [`Hardswish`](nn/_autosummary/mlx.nn.Hardswish.html#mlx.nn.Hardswish)
  + [mlx.nn.InstanceNorm](nn/_autosummary/mlx.nn.InstanceNorm.html)
    - [`InstanceNorm`](nn/_autosummary/mlx.nn.InstanceNorm.html#mlx.nn.InstanceNorm)
  + [mlx.nn.LayerNorm](nn/_autosummary/mlx.nn.LayerNorm.html)
    - [`LayerNorm`](nn/_autosummary/mlx.nn.LayerNorm.html#mlx.nn.LayerNorm)
  + [mlx.nn.LeakyReLU](nn/_autosummary/mlx.nn.LeakyReLU.html)
    - [`LeakyReLU`](nn/_autosummary/mlx.nn.LeakyReLU.html#mlx.nn.LeakyReLU)
  + [mlx.nn.Linear](nn/_autosummary/mlx.nn.Linear.html)
    - [`Linear`](nn/_autosummary/mlx.nn.Linear.html#mlx.nn.Linear)
  + [mlx.nn.LogSigmoid](nn/_autosummary/mlx.nn.LogSigmoid.html)
    - [`LogSigmoid`](nn/_autosummary/mlx.nn.LogSigmoid.html#mlx.nn.LogSigmoid)
  + [mlx.nn.LogSoftmax](nn/_autosummary/mlx.nn.LogSoftmax.html)
    - [`LogSoftmax`](nn/_autosummary/mlx.nn.LogSoftmax.html#mlx.nn.LogSoftmax)
  + [mlx.nn.LSTM](nn/_autosummary/mlx.nn.LSTM.html)
    - [`LSTM`](nn/_autosummary/mlx.nn.LSTM.html#mlx.nn.LSTM)
  + [mlx.nn.MaxPool1d](nn/_autosummary/mlx.nn.MaxPool1d.html)
    - [`MaxPool1d`](nn/_autosummary/mlx.nn.MaxPool1d.html#mlx.nn.MaxPool1d)
  + [mlx.nn.MaxPool2d](nn/_autosummary/mlx.nn.MaxPool2d.html)
    - [`MaxPool2d`](nn/_autosummary/mlx.nn.MaxPool2d.html#mlx.nn.MaxPool2d)
  + [mlx.nn.MaxPool3d](nn/_autosummary/mlx.nn.MaxPool3d.html)
    - [`MaxPool3d`](nn/_autosummary/mlx.nn.MaxPool3d.html#mlx.nn.MaxPool3d)
  + [mlx.nn.Mish](nn/_autosummary/mlx.nn.Mish.html)
    - [`Mish`](nn/_autosummary/mlx.nn.Mish.html#mlx.nn.Mish)
  + [mlx.nn.MultiHeadAttention](nn/_autosummary/mlx.nn.MultiHeadAttention.html)
    - [`MultiHeadAttention`](nn/_autosummary/mlx.nn.MultiHeadAttention.html#mlx.nn.MultiHeadAttention)
  + [mlx.nn.PReLU](nn/_autosummary/mlx.nn.PReLU.html)
    - [`PReLU`](nn/_autosummary/mlx.nn.PReLU.html#mlx.nn.PReLU)
  + [mlx.nn.QuantizedEmbedding](nn/_autosummary/mlx.nn.QuantizedEmbedding.html)
    - [`QuantizedEmbedding`](nn/_autosummary/mlx.nn.QuantizedEmbedding.html#mlx.nn.QuantizedEmbedding)
  + [mlx.nn.QuantizedLinear](nn/_autosummary/mlx.nn.QuantizedLinear.html)
    - [`QuantizedLinear`](nn/_autosummary/mlx.nn.QuantizedLinear.html#mlx.nn.QuantizedLinear)
  + [mlx.nn.RMSNorm](nn/_autosummary/mlx.nn.RMSNorm.html)
    - [`RMSNorm`](nn/_autosummary/mlx.nn.RMSNorm.html#mlx.nn.RMSNorm)
  + [mlx.nn.ReLU](nn/_autosummary/mlx.nn.ReLU.html)
    - [`ReLU`](nn/_autosummary/mlx.nn.ReLU.html#mlx.nn.ReLU)
  + [mlx.nn.ReLU2](nn/_autosummary/mlx.nn.ReLU2.html)
    - [`ReLU2`](nn/_autosummary/mlx.nn.ReLU2.html#mlx.nn.ReLU2)
  + [mlx.nn.ReLU6](nn/_autosummary/mlx.nn.ReLU6.html)
    - [`ReLU6`](nn/_autosummary/mlx.nn.ReLU6.html#mlx.nn.ReLU6)
  + [mlx.nn.RNN](nn/_autosummary/mlx.nn.RNN.html)
    - [`RNN`](nn/_autosummary/mlx.nn.RNN.html#mlx.nn.RNN)
  + [mlx.nn.RoPE](nn/_autosummary/mlx.nn.RoPE.html)
    - [`RoPE`](nn/_autosummary/mlx.nn.RoPE.html#mlx.nn.RoPE)
  + [mlx.nn.SELU](nn/_autosummary/mlx.nn.SELU.html)
    - [`SELU`](nn/_autosummary/mlx.nn.SELU.html#mlx.nn.SELU)
  + [mlx.nn.Sequential](nn/_autosummary/mlx.nn.Sequential.html)
    - [`Sequential`](nn/_autosummary/mlx.nn.Sequential.html#mlx.nn.Sequential)
  + [mlx.nn.Sigmoid](nn/_autosummary/mlx.nn.Sigmoid.html)
    - [`Sigmoid`](nn/_autosummary/mlx.nn.Sigmoid.html#mlx.nn.Sigmoid)
  + [mlx.nn.SiLU](nn/_autosummary/mlx.nn.SiLU.html)
    - [`SiLU`](nn/_autosummary/mlx.nn.SiLU.html#mlx.nn.SiLU)
  + [mlx.nn.SinusoidalPositionalEncoding](nn/_autosummary/mlx.nn.SinusoidalPositionalEncoding.html)
    - [`SinusoidalPositionalEncoding`](nn/_autosummary/mlx.nn.SinusoidalPositionalEncoding.html#mlx.nn.SinusoidalPositionalEncoding)
  + [mlx.nn.Softmin](nn/_autosummary/mlx.nn.Softmin.html)
    - [`Softmin`](nn/_autosummary/mlx.nn.Softmin.html#mlx.nn.Softmin)
  + [mlx.nn.Softshrink](nn/_autosummary/mlx.nn.Softshrink.html)
    - [`Softshrink`](nn/_autosummary/mlx.nn.Softshrink.html#mlx.nn.Softshrink)
  + [mlx.nn.Softsign](nn/_autosummary/mlx.nn.Softsign.html)
    - [`Softsign`](nn/_autosummary/mlx.nn.Softsign.html#mlx.nn.Softsign)
  + [mlx.nn.Softmax](nn/_autosummary/mlx.nn.Softmax.html)
    - [`Softmax`](nn/_autosummary/mlx.nn.Softmax.html#mlx.nn.Softmax)
  + [mlx.nn.Softplus](nn/_autosummary/mlx.nn.Softplus.html)
    - [`Softplus`](nn/_autosummary/mlx.nn.Softplus.html#mlx.nn.Softplus)
  + [mlx.nn.Step](nn/_autosummary/mlx.nn.Step.html)
    - [`Step`](nn/_autosummary/mlx.nn.Step.html#mlx.nn.Step)
  + [mlx.nn.Tanh](nn/_autosummary/mlx.nn.Tanh.html)
    - [`Tanh`](nn/_autosummary/mlx.nn.Tanh.html#mlx.nn.Tanh)
  + [mlx.nn.Transformer](nn/_autosummary/mlx.nn.Transformer.html)
    - [`Transformer`](nn/_autosummary/mlx.nn.Transformer.html#mlx.nn.Transformer)
  + [mlx.nn.Upsample](nn/_autosummary/mlx.nn.Upsample.html)
    - [`Upsample`](nn/_autosummary/mlx.nn.Upsample.html#mlx.nn.Upsample)
* [Functions](nn/functions.html)
  + [mlx.nn.elu](nn/_autosummary_functions/mlx.nn.elu.html)
    - [`elu`](nn/_autosummary_functions/mlx.nn.elu.html#mlx.nn.elu)
  + [mlx.nn.celu](nn/_autosummary_functions/mlx.nn.celu.html)
    - [`celu`](nn/_autosummary_functions/mlx.nn.celu.html#mlx.nn.celu)
  + [mlx.nn.gelu](nn/_autosummary_functions/mlx.nn.gelu.html)
    - [`gelu`](nn/_autosummary_functions/mlx.nn.gelu.html#mlx.nn.gelu)
  + [mlx.nn.gelu\_approx](nn/_autosummary_functions/mlx.nn.gelu_approx.html)
    - [`gelu_approx`](nn/_autosummary_functions/mlx.nn.gelu_approx.html#mlx.nn.gelu_approx)
  + [mlx.nn.gelu\_fast\_approx](nn/_autosummary_functions/mlx.nn.gelu_fast_approx.html)
    - [`gelu_fast_approx`](nn/_autosummary_functions/mlx.nn.gelu_fast_approx.html#mlx.nn.gelu_fast_approx)
  + [mlx.nn.glu](nn/_autosummary_functions/mlx.nn.glu.html)
    - [`glu`](nn/_autosummary_functions/mlx.nn.glu.html#mlx.nn.glu)
  + [mlx.nn.hard\_shrink](nn/_autosummary_functions/mlx.nn.hard_shrink.html)
    - [`hard_shrink`](nn/_autosummary_functions/mlx.nn.hard_shrink.html#mlx.nn.hard_shrink)
  + [mlx.nn.hard\_tanh](nn/_autosummary_functions/mlx.nn.hard_tanh.html)
    - [`hard_tanh`](nn/_autosummary_functions/mlx.nn.hard_tanh.html#mlx.nn.hard_tanh)
  + [mlx.nn.hardswish](nn/_autosummary_functions/mlx.nn.hardswish.html)
    - [`hardswish`](nn/_autosummary_functions/mlx.nn.hardswish.html#mlx.nn.hardswish)
  + [mlx.nn.leaky\_relu](nn/_autosummary_functions/mlx.nn.leaky_relu.html)
    - [`leaky_relu`](nn/_autosummary_functions/mlx.nn.leaky_relu.html#mlx.nn.leaky_relu)
  + [mlx.nn.log\_sigmoid](nn/_autosummary_functions/mlx.nn.log_sigmoid.html)
    - [`log_sigmoid`](nn/_autosummary_functions/mlx.nn.log_sigmoid.html#mlx.nn.log_sigmoid)
  + [mlx.nn.log\_softmax](nn/_autosummary_functions/mlx.nn.log_softmax.html)
    - [`log_softmax`](nn/_autosummary_functions/mlx.nn.log_softmax.html#mlx.nn.log_softmax)
  + [mlx.nn.mish](nn/_autosummary_functions/mlx.nn.mish.html)
    - [`mish`](nn/_autosummary_functions/mlx.nn.mish.html#mlx.nn.mish)
  + [mlx.nn.prelu](nn/_autosummary_functions/mlx.nn.prelu.html)
    - [`prelu`](nn/_autosummary_functions/mlx.nn.prelu.html#mlx.nn.prelu)
  + [mlx.nn.relu](nn/_autosummary_functions/mlx.nn.relu.html)
    - [`relu`](nn/_autosummary_functions/mlx.nn.relu.html#mlx.nn.relu)
  + [mlx.nn.relu2](nn/_autosummary_functions/mlx.nn.relu2.html)
    - [`relu2`](nn/_autosummary_functions/mlx.nn.relu2.html#mlx.nn.relu2)
  + [mlx.nn.relu6](nn/_autosummary_functions/mlx.nn.relu6.html)
    - [`relu6`](nn/_autosummary_functions/mlx.nn.relu6.html#mlx.nn.relu6)
  + [mlx.nn.selu](nn/_autosummary_functions/mlx.nn.selu.html)
    - [`selu`](nn/_autosummary_functions/mlx.nn.selu.html#mlx.nn.selu)
  + [mlx.nn.sigmoid](nn/_autosummary_functions/mlx.nn.sigmoid.html)
    - [`sigmoid`](nn/_autosummary_functions/mlx.nn.sigmoid.html#mlx.nn.sigmoid)
  + [mlx.nn.silu](nn/_autosummary_functions/mlx.nn.silu.html)
    - [`silu`](nn/_autosummary_functions/mlx.nn.silu.html#mlx.nn.silu)
  + [mlx.nn.softmax](nn/_autosummary_functions/mlx.nn.softmax.html)
    - [`softmax`](nn/_autosummary_functions/mlx.nn.softmax.html#mlx.nn.softmax)
  + [mlx.nn.softmin](nn/_autosummary_functions/mlx.nn.softmin.html)
    - [`softmin`](nn/_autosummary_functions/mlx.nn.softmin.html#mlx.nn.softmin)
  + [mlx.nn.softplus](nn/_autosummary_functions/mlx.nn.softplus.html)
    - [`softplus`](nn/_autosummary_functions/mlx.nn.softplus.html#mlx.nn.softplus)
  + [mlx.nn.softshrink](nn/_autosummary_functions/mlx.nn.softshrink.html)
    - [`softshrink`](nn/_autosummary_functions/mlx.nn.softshrink.html#mlx.nn.softshrink)
  + [mlx.nn.step](nn/_autosummary_functions/mlx.nn.step.html)
    - [`step`](nn/_autosummary_functions/mlx.nn.step.html#mlx.nn.step)
  + [mlx.nn.tanh](nn/_autosummary_functions/mlx.nn.tanh.html)
    - [`tanh`](nn/_autosummary_functions/mlx.nn.tanh.html#mlx.nn.tanh)
* [Loss Functions](nn/losses.html)
  + [mlx.nn.losses.binary\_cross\_entropy](nn/_autosummary_functions/mlx.nn.losses.binary_cross_entropy.html)
    - [`binary_cross_entropy`](nn/_autosummary_functions/mlx.nn.losses.binary_cross_entropy.html#mlx.nn.losses.binary_cross_entropy)
  + [mlx.nn.losses.cosine\_similarity\_loss](nn/_autosummary_functions/mlx.nn.losses.cosine_similarity_loss.html)
    - [`cosine_similarity_loss`](nn/_autosummary_functions/mlx.nn.losses.cosine_similarity_loss.html#mlx.nn.losses.cosine_similarity_loss)
  + [mlx.nn.losses.cross\_entropy](nn/_autosummary_functions/mlx.nn.losses.cross_entropy.html)
    - [`cross_entropy`](nn/_autosummary_functions/mlx.nn.losses.cross_entropy.html#mlx.nn.losses.cross_entropy)
  + [mlx.nn.losses.gaussian\_nll\_loss](nn/_autosummary_functions/mlx.nn.losses.gaussian_nll_loss.html)
    - [`gaussian_nll_loss`](nn/_autosummary_functions/mlx.nn.losses.gaussian_nll_loss.html#mlx.nn.losses.gaussian_nll_loss)
  + [mlx.nn.losses.hinge\_loss](nn/_autosummary_functions/mlx.nn.losses.hinge_loss.html)
    - [`hinge_loss`](nn/_autosummary_functions/mlx.nn.losses.hinge_loss.html#mlx.nn.losses.hinge_loss)
  + [mlx.nn.losses.huber\_loss](nn/_autosummary_functions/mlx.nn.losses.huber_loss.html)
    - [`huber_loss`](nn/_autosummary_functions/mlx.nn.losses.huber_loss.html#mlx.nn.losses.huber_loss)
  + [mlx.nn.losses.kl\_div\_loss](nn/_autosummary_functions/mlx.nn.losses.kl_div_loss.html)
    - [`kl_div_loss`](nn/_autosummary_functions/mlx.nn.losses.kl_div_loss.html#mlx.nn.losses.kl_div_loss)
  + [mlx.nn.losses.l1\_loss](nn/_autosummary_functions/mlx.nn.losses.l1_loss.html)
    - [`l1_loss`](nn/_autosummary_functions/mlx.nn.losses.l1_loss.html#mlx.nn.losses.l1_loss)
  + [mlx.nn.losses.log\_cosh\_loss](nn/_autosummary_functions/mlx.nn.losses.log_cosh_loss.html)
    - [`log_cosh_loss`](nn/_autosummary_functions/mlx.nn.losses.log_cosh_loss.html#mlx.nn.losses.log_cosh_loss)
  + [mlx.nn.losses.margin\_ranking\_loss](nn/_autosummary_functions/mlx.nn.losses.margin_ranking_loss.html)
    - [`margin_ranking_loss`](nn/_autosummary_functions/mlx.nn.losses.margin_ranking_loss.html#mlx.nn.losses.margin_ranking_loss)
  + [mlx.nn.losses.mse\_loss](nn/_autosummary_functions/mlx.nn.losses.mse_loss.html)
    - [`mse_loss`](nn/_autosummary_functions/mlx.nn.losses.mse_loss.html#mlx.nn.losses.mse_loss)
  + [mlx.nn.losses.nll\_loss](nn/_autosummary_functions/mlx.nn.losses.nll_loss.html)
    - [`nll_loss`](nn/_autosummary_functions/mlx.nn.losses.nll_loss.html#mlx.nn.losses.nll_loss)
  + [mlx.nn.losses.smooth\_l1\_loss](nn/_autosummary_functions/mlx.nn.losses.smooth_l1_loss.html)
    - [`smooth_l1_loss`](nn/_autosummary_functions/mlx.nn.losses.smooth_l1_loss.html#mlx.nn.losses.smooth_l1_loss)
  + [mlx.nn.losses.triplet\_loss](nn/_autosummary_functions/mlx.nn.losses.triplet_loss.html)
    - [`triplet_loss`](nn/_autosummary_functions/mlx.nn.losses.triplet_loss.html#mlx.nn.losses.triplet_loss)
* [Initializers](nn/init.html)
  + [mlx.nn.init.constant](nn/_autosummary/mlx.nn.init.constant.html)
    - [`constant()`](nn/_autosummary/mlx.nn.init.constant.html#mlx.nn.init.constant)
  + [mlx.nn.init.normal](nn/_autosummary/mlx.nn.init.normal.html)
    - [`normal()`](nn/_autosummary/mlx.nn.init.normal.html#mlx.nn.init.normal)
  + [mlx.nn.init.uniform](nn/_autosummary/mlx.nn.init.uniform.html)
    - [`uniform()`](nn/_autosummary/mlx.nn.init.uniform.html#mlx.nn.init.uniform)
  + [mlx.nn.init.identity](nn/_autosummary/mlx.nn.init.identity.html)
    - [`identity()`](nn/_autosummary/mlx.nn.init.identity.html#mlx.nn.init.identity)
  + [mlx.nn.init.glorot\_normal](nn/_autosummary/mlx.nn.init.glorot_normal.html)
    - [`glorot_normal()`](nn/_autosummary/mlx.nn.init.glorot_normal.html#mlx.nn.init.glorot_normal)
  + [mlx.nn.init.glorot\_uniform](nn/_autosummary/mlx.nn.init.glorot_uniform.html)
    - [`glorot_uniform()`](nn/_autosummary/mlx.nn.init.glorot_uniform.html#mlx.nn.init.glorot_uniform)
  + [mlx.nn.init.he\_normal](nn/_autosummary/mlx.nn.init.he_normal.html)
    - [`he_normal()`](nn/_autosummary/mlx.nn.init.he_normal.html#mlx.nn.init.he_normal)
  + [mlx.nn.init.he\_uniform](nn/_autosummary/mlx.nn.init.he_uniform.html)
    - [`he_uniform()`](nn/_autosummary/mlx.nn.init.he_uniform.html#mlx.nn.init.he_uniform)

Contents

---

# mlx_nn_functions

Source: https://ml-explore.github.io/mlx/build/html/python/nn/functions.html

---

* [.rst](../../_sources/python/nn/functions.rst)
* .pdf

# Functions

# Functions[#](#functions "Link to this heading")

Layers without parameters (e.g. activation functions) are also provided as
simple functions.

|  |  |
| --- | --- |
| [`elu`](_autosummary_functions/mlx.nn.elu.html#mlx.nn.elu "mlx.nn.elu")(x[, alpha]) | Applies the Exponential Linear Unit. |
| [`celu`](_autosummary_functions/mlx.nn.celu.html#mlx.nn.celu "mlx.nn.celu")(x[, alpha]) | Applies the Continuously Differentiable Exponential Linear Unit. |
| [`gelu`](_autosummary_functions/mlx.nn.gelu.html#mlx.nn.gelu "mlx.nn.gelu")(x) | Applies the Gaussian Error Linear Units function. |
| [`gelu_approx`](_autosummary_functions/mlx.nn.gelu_approx.html#mlx.nn.gelu_approx "mlx.nn.gelu_approx")(x) | An approximation to Gaussian Error Linear Unit. |
| [`gelu_fast_approx`](_autosummary_functions/mlx.nn.gelu_fast_approx.html#mlx.nn.gelu_fast_approx "mlx.nn.gelu_fast_approx")(x) | A fast approximation to Gaussian Error Linear Unit. |
| [`glu`](_autosummary_functions/mlx.nn.glu.html#mlx.nn.glu "mlx.nn.glu")(x[, axis]) | Applies the gated linear unit function. |
| [`hard_shrink`](_autosummary_functions/mlx.nn.hard_shrink.html#mlx.nn.hard_shrink "mlx.nn.hard_shrink")(x[, lambd]) | Applies the HardShrink activation function. |
| [`hard_tanh`](_autosummary_functions/mlx.nn.hard_tanh.html#mlx.nn.hard_tanh "mlx.nn.hard_tanh")(x[, min\_val, max\_val]) | Applies the HardTanh function. |
| [`hardswish`](_autosummary_functions/mlx.nn.hardswish.html#mlx.nn.hardswish "mlx.nn.hardswish")(x) | Applies the hardswish function, element-wise. |
| [`leaky_relu`](_autosummary_functions/mlx.nn.leaky_relu.html#mlx.nn.leaky_relu "mlx.nn.leaky_relu")(x[, negative\_slope]) | Applies the Leaky Rectified Linear Unit. |
| [`log_sigmoid`](_autosummary_functions/mlx.nn.log_sigmoid.html#mlx.nn.log_sigmoid "mlx.nn.log_sigmoid")(x) | Applies the Log Sigmoid function. |
| [`log_softmax`](_autosummary_functions/mlx.nn.log_softmax.html#mlx.nn.log_softmax "mlx.nn.log_softmax")(x[, axis]) | Applies the Log Softmax function. |
| [`mish`](_autosummary_functions/mlx.nn.mish.html#mlx.nn.mish "mlx.nn.mish")(x) | Applies the Mish function, element-wise. |
| [`prelu`](_autosummary_functions/mlx.nn.prelu.html#mlx.nn.prelu "mlx.nn.prelu")(x, alpha) | Applies the element-wise parametric ReLU. |
| [`relu`](_autosummary_functions/mlx.nn.relu.html#mlx.nn.relu "mlx.nn.relu")(x) | Applies the Rectified Linear Unit. |
| [`relu2`](_autosummary_functions/mlx.nn.relu2.html#mlx.nn.relu2 "mlx.nn.relu2")(x) | Applies the ReLU² activation function. |
| [`relu6`](_autosummary_functions/mlx.nn.relu6.html#mlx.nn.relu6 "mlx.nn.relu6")(x) | Applies the Rectified Linear Unit 6. |
| [`selu`](_autosummary_functions/mlx.nn.selu.html#mlx.nn.selu "mlx.nn.selu")(x) | Applies the Scaled Exponential Linear Unit. |
| [`sigmoid`](_autosummary_functions/mlx.nn.sigmoid.html#mlx.nn.sigmoid "mlx.nn.sigmoid")(x) | Applies the sigmoid function. |
| [`silu`](_autosummary_functions/mlx.nn.silu.html#mlx.nn.silu "mlx.nn.silu")(x) | Applies the Sigmoid Linear Unit. |
| [`softmax`](_autosummary_functions/mlx.nn.softmax.html#mlx.nn.softmax "mlx.nn.softmax")(x[, axis]) | Applies the Softmax function. |
| [`softmin`](_autosummary_functions/mlx.nn.softmin.html#mlx.nn.softmin "mlx.nn.softmin")(x[, axis]) | Applies the Softmin function. |
| [`softplus`](_autosummary_functions/mlx.nn.softplus.html#mlx.nn.softplus "mlx.nn.softplus")(x) | Applies the Softplus function. |
| [`softshrink`](_autosummary_functions/mlx.nn.softshrink.html#mlx.nn.softshrink "mlx.nn.softshrink")(x[, lambd]) | Applies the Softshrink activation function. |
| [`step`](_autosummary_functions/mlx.nn.step.html#mlx.nn.step "mlx.nn.step")(x[, threshold]) | Applies the Step Activation Function. |
| [`tanh`](_autosummary_functions/mlx.nn.tanh.html#mlx.nn.tanh "mlx.nn.tanh")(x) | Applies the hyperbolic tangent function. |

---

# mlx_nn_layers_detail

Source: https://ml-explore.github.io/mlx/build/html/python/nn/layers.html

---

* [.rst](../../_sources/python/nn/layers.rst)
* .pdf

# Layers

# Layers[#](#layers "Link to this heading")

|  |  |
| --- | --- |
| [`ALiBi`](_autosummary/mlx.nn.ALiBi.html#mlx.nn.ALiBi "mlx.nn.ALiBi")() |  |
| [`AvgPool1d`](_autosummary/mlx.nn.AvgPool1d.html#mlx.nn.AvgPool1d "mlx.nn.AvgPool1d")(kernel\_size[, stride, padding]) | Applies 1-dimensional average pooling. |
| [`AvgPool2d`](_autosummary/mlx.nn.AvgPool2d.html#mlx.nn.AvgPool2d "mlx.nn.AvgPool2d")(kernel\_size[, stride, padding]) | Applies 2-dimensional average pooling. |
| [`AvgPool3d`](_autosummary/mlx.nn.AvgPool3d.html#mlx.nn.AvgPool3d "mlx.nn.AvgPool3d")(kernel\_size[, stride, padding]) | Applies 3-dimensional average pooling. |
| [`BatchNorm`](_autosummary/mlx.nn.BatchNorm.html#mlx.nn.BatchNorm "mlx.nn.BatchNorm")(num\_features[, eps, momentum, ...]) | Applies Batch Normalization over a 2D or 3D input. |
| [`CELU`](_autosummary/mlx.nn.CELU.html#mlx.nn.CELU "mlx.nn.CELU")([alpha]) | Applies the Continuously Differentiable Exponential Linear Unit. |
| [`Conv1d`](_autosummary/mlx.nn.Conv1d.html#mlx.nn.Conv1d "mlx.nn.Conv1d")(in\_channels, out\_channels, kernel\_size) | Applies a 1-dimensional convolution over the multi-channel input sequence. |
| [`Conv2d`](_autosummary/mlx.nn.Conv2d.html#mlx.nn.Conv2d "mlx.nn.Conv2d")(in\_channels, out\_channels, kernel\_size) | Applies a 2-dimensional convolution over the multi-channel input image. |
| [`Conv3d`](_autosummary/mlx.nn.Conv3d.html#mlx.nn.Conv3d "mlx.nn.Conv3d")(in\_channels, out\_channels, kernel\_size) | Applies a 3-dimensional convolution over the multi-channel input image. |
| [`ConvTranspose1d`](_autosummary/mlx.nn.ConvTranspose1d.html#mlx.nn.ConvTranspose1d "mlx.nn.ConvTranspose1d")(in\_channels, out\_channels, ...) | Applies a 1-dimensional transposed convolution over the multi-channel input sequence. |
| [`ConvTranspose2d`](_autosummary/mlx.nn.ConvTranspose2d.html#mlx.nn.ConvTranspose2d "mlx.nn.ConvTranspose2d")(in\_channels, out\_channels, ...) | Applies a 2-dimensional transposed convolution over the multi-channel input image. |
| [`ConvTranspose3d`](_autosummary/mlx.nn.ConvTranspose3d.html#mlx.nn.ConvTranspose3d "mlx.nn.ConvTranspose3d")(in\_channels, out\_channels, ...) | Applies a 3-dimensional transposed convolution over the multi-channel input image. |
| [`Dropout`](_autosummary/mlx.nn.Dropout.html#mlx.nn.Dropout "mlx.nn.Dropout")([p]) | Randomly zero a portion of the elements during training. |
| [`Dropout2d`](_autosummary/mlx.nn.Dropout2d.html#mlx.nn.Dropout2d "mlx.nn.Dropout2d")([p]) | Apply 2D channel-wise dropout during training. |
| [`Dropout3d`](_autosummary/mlx.nn.Dropout3d.html#mlx.nn.Dropout3d "mlx.nn.Dropout3d")([p]) | Apply 3D channel-wise dropout during training. |
| [`Embedding`](_autosummary/mlx.nn.Embedding.html#mlx.nn.Embedding "mlx.nn.Embedding")(num\_embeddings, dims) | Implements a simple lookup table that maps each input integer to a high-dimensional vector. |
| [`ELU`](_autosummary/mlx.nn.ELU.html#mlx.nn.ELU "mlx.nn.ELU")([alpha]) | Applies the Exponential Linear Unit. |
| [`GELU`](_autosummary/mlx.nn.GELU.html#mlx.nn.GELU "mlx.nn.GELU")([approx]) | Applies the Gaussian Error Linear Units. |
| [`GLU`](_autosummary/mlx.nn.GLU.html#mlx.nn.GLU "mlx.nn.GLU")([axis]) | Applies the gated linear unit function. |
| [`GroupNorm`](_autosummary/mlx.nn.GroupNorm.html#mlx.nn.GroupNorm "mlx.nn.GroupNorm")(num\_groups, dims[, eps, affine, ...]) | Applies Group Normalization [1] to the inputs. |
| [`GRU`](_autosummary/mlx.nn.GRU.html#mlx.nn.GRU "mlx.nn.GRU")(input\_size, hidden\_size[, bias]) | A gated recurrent unit (GRU) RNN layer. |
| [`HardShrink`](_autosummary/mlx.nn.HardShrink.html#mlx.nn.HardShrink "mlx.nn.HardShrink")() | Applies the HardShrink function. |
| [`HardTanh`](_autosummary/mlx.nn.HardTanh.html#mlx.nn.HardTanh "mlx.nn.HardTanh")() | Applies the HardTanh function. |
| [`Hardswish`](_autosummary/mlx.nn.Hardswish.html#mlx.nn.Hardswish "mlx.nn.Hardswish")() | Applies the hardswish function, element-wise. |
| [`InstanceNorm`](_autosummary/mlx.nn.InstanceNorm.html#mlx.nn.InstanceNorm "mlx.nn.InstanceNorm")(dims[, eps, affine]) | Applies instance normalization [1] on the inputs. |
| [`LayerNorm`](_autosummary/mlx.nn.LayerNorm.html#mlx.nn.LayerNorm "mlx.nn.LayerNorm")(dims[, eps, affine, bias]) | Applies layer normalization [1] on the inputs. |
| [`LeakyReLU`](_autosummary/mlx.nn.LeakyReLU.html#mlx.nn.LeakyReLU "mlx.nn.LeakyReLU")([negative\_slope]) | Applies the Leaky Rectified Linear Unit. |
| [`Linear`](_autosummary/mlx.nn.Linear.html#mlx.nn.Linear "mlx.nn.Linear")(input\_dims, output\_dims[, bias]) | Applies an affine transformation to the input. |
| [`LogSigmoid`](_autosummary/mlx.nn.LogSigmoid.html#mlx.nn.LogSigmoid "mlx.nn.LogSigmoid")() | Applies the Log Sigmoid function. |
| [`LogSoftmax`](_autosummary/mlx.nn.LogSoftmax.html#mlx.nn.LogSoftmax "mlx.nn.LogSoftmax")() | Applies the Log Softmax function. |
| [`LSTM`](_autosummary/mlx.nn.LSTM.html#mlx.nn.LSTM "mlx.nn.LSTM")(input\_size, hidden\_size[, bias]) | An LSTM recurrent layer. |
| [`MaxPool1d`](_autosummary/mlx.nn.MaxPool1d.html#mlx.nn.MaxPool1d "mlx.nn.MaxPool1d")(kernel\_size[, stride, padding]) | Applies 1-dimensional max pooling. |
| [`MaxPool2d`](_autosummary/mlx.nn.MaxPool2d.html#mlx.nn.MaxPool2d "mlx.nn.MaxPool2d")(kernel\_size[, stride, padding]) | Applies 2-dimensional max pooling. |
| [`MaxPool3d`](_autosummary/mlx.nn.MaxPool3d.html#mlx.nn.MaxPool3d "mlx.nn.MaxPool3d")(kernel\_size[, stride, padding]) | Applies 3-dimensional max pooling. |
| [`Mish`](_autosummary/mlx.nn.Mish.html#mlx.nn.Mish "mlx.nn.Mish")() | Applies the Mish function, element-wise. |
| [`MultiHeadAttention`](_autosummary/mlx.nn.MultiHeadAttention.html#mlx.nn.MultiHeadAttention "mlx.nn.MultiHeadAttention")(dims, num\_heads[, ...]) | Implements the scaled dot product attention with multiple heads. |
| [`PReLU`](_autosummary/mlx.nn.PReLU.html#mlx.nn.PReLU "mlx.nn.PReLU")([num\_parameters, init]) | Applies the element-wise parametric ReLU. |
| [`QuantizedEmbedding`](_autosummary/mlx.nn.QuantizedEmbedding.html#mlx.nn.QuantizedEmbedding "mlx.nn.QuantizedEmbedding")(num\_embeddings, dims[, ...]) | The same as [`Embedding`](_autosummary/mlx.nn.Embedding.html#mlx.nn.Embedding "mlx.nn.Embedding") but with a quantized weight matrix. |
| [`QuantizedLinear`](_autosummary/mlx.nn.QuantizedLinear.html#mlx.nn.QuantizedLinear "mlx.nn.QuantizedLinear")(input\_dims, output\_dims[, ...]) | Applies an affine transformation to the input using a quantized weight matrix. |
| [`RMSNorm`](_autosummary/mlx.nn.RMSNorm.html#mlx.nn.RMSNorm "mlx.nn.RMSNorm")(dims[, eps]) | Applies Root Mean Square normalization [1] to the inputs. |
| [`ReLU`](_autosummary/mlx.nn.ReLU.html#mlx.nn.ReLU "mlx.nn.ReLU")() | Applies the Rectified Linear Unit. |
| [`ReLU2`](_autosummary/mlx.nn.ReLU2.html#mlx.nn.ReLU2 "mlx.nn.ReLU2")() | Applies the ReLU² activation function. |
| [`ReLU6`](_autosummary/mlx.nn.ReLU6.html#mlx.nn.ReLU6 "mlx.nn.ReLU6")() | Applies the Rectified Linear Unit 6. |
| [`RNN`](_autosummary/mlx.nn.RNN.html#mlx.nn.RNN "mlx.nn.RNN")(input\_size, hidden\_size[, bias, ...]) | An Elman recurrent layer. |
| [`RoPE`](_autosummary/mlx.nn.RoPE.html#mlx.nn.RoPE "mlx.nn.RoPE")(dims[, traditional, base, scale]) | Implements the rotary positional encoding. |
| [`SELU`](_autosummary/mlx.nn.SELU.html#mlx.nn.SELU "mlx.nn.SELU")() | Applies the Scaled Exponential Linear Unit. |
| [`Sequential`](_autosummary/mlx.nn.Sequential.html#mlx.nn.Sequential "mlx.nn.Sequential")(\*modules) | A layer that calls the passed callables in order. |
| [`Sigmoid`](_autosummary/mlx.nn.Sigmoid.html#mlx.nn.Sigmoid "mlx.nn.Sigmoid")() | Applies the sigmoid function, element-wise. |
| [`SiLU`](_autosummary/mlx.nn.SiLU.html#mlx.nn.SiLU "mlx.nn.SiLU")() | Applies the Sigmoid Linear Unit. |
| [`SinusoidalPositionalEncoding`](_autosummary/mlx.nn.SinusoidalPositionalEncoding.html#mlx.nn.SinusoidalPositionalEncoding "mlx.nn.SinusoidalPositionalEncoding")(dims[, ...]) | Implements sinusoidal positional encoding. |
| [`Softmin`](_autosummary/mlx.nn.Softmin.html#mlx.nn.Softmin "mlx.nn.Softmin")() | Applies the Softmin function. |
| [`Softshrink`](_autosummary/mlx.nn.Softshrink.html#mlx.nn.Softshrink "mlx.nn.Softshrink")([lambd]) | Applies the Softshrink function. |
| [`Softsign`](_autosummary/mlx.nn.Softsign.html#mlx.nn.Softsign "mlx.nn.Softsign")() | Applies the Softsign function. |
| [`Softmax`](_autosummary/mlx.nn.Softmax.html#mlx.nn.Softmax "mlx.nn.Softmax")() | Applies the Softmax function. |
| [`Softplus`](_autosummary/mlx.nn.Softplus.html#mlx.nn.Softplus "mlx.nn.Softplus")() | Applies the Softplus function. |
| [`Step`](_autosummary/mlx.nn.Step.html#mlx.nn.Step "mlx.nn.Step")([threshold]) | Applies the Step Activation Function. |
| [`Tanh`](_autosummary/mlx.nn.Tanh.html#mlx.nn.Tanh "mlx.nn.Tanh")() | Applies the hyperbolic tangent function. |
| [`Transformer`](_autosummary/mlx.nn.Transformer.html#mlx.nn.Transformer "mlx.nn.Transformer")(dims, num\_heads, ...) | Implements a standard Transformer model. |
| [`Upsample`](_autosummary/mlx.nn.Upsample.html#mlx.nn.Upsample "mlx.nn.Upsample")(scale\_factor[, mode, align\_corners]) | Upsample the input signal spatially. |

---

# mlx_llama_inference

Source: https://ml-explore.github.io/mlx/build/html/examples/llama-inference.html

---

* [.rst](../_sources/examples/llama-inference.rst)
* .pdf

# LLM inference

## Contents

# LLM inference[#](#llm-inference "Link to this heading")

MLX enables efficient inference of large-ish transformers on Apple silicon
without compromising on ease of use. In this example we will create an
inference script for the Llama family of transformer models in which the model
is defined in less than 200 lines of python.

## Implementing the model[#](#implementing-the-model "Link to this heading")

We will use the neural network building blocks defined in the `mlx.nn`
module to concisely define the model architecture.

### Attention layer[#](#attention-layer "Link to this heading")

We will start with the Llama attention layer which notably uses the RoPE
positional encoding. [[1]](#id4) In addition, our attention layer will optionally use a
key/value cache that will be concatenated with the provided keys and values to
support efficient inference.

Our implementation uses [`mlx.nn.Linear`](../python/nn/_autosummary/mlx.nn.Linear.html#mlx.nn.Linear "mlx.nn.Linear") for all the projections and
[`mlx.nn.RoPE`](../python/nn/_autosummary/mlx.nn.RoPE.html#mlx.nn.RoPE "mlx.nn.RoPE") for the positional encoding.

```python
import mlx.core as mx
import mlx.nn as nn

class LlamaAttention(nn.Module):
    def __init__(self, dims: int, num_heads: int):
        super().__init__()

        self.num_heads = num_heads

        self.rope = nn.RoPE(dims // num_heads, traditional=True)
        self.query_proj = nn.Linear(dims, dims, bias=False)
        self.key_proj = nn.Linear(dims, dims, bias=False)
        self.value_proj = nn.Linear(dims, dims, bias=False)
        self.out_proj = nn.Linear(dims, dims, bias=False)

    def __call__(self, queries, keys, values, mask=None, cache=None):
        queries = self.query_proj(queries)
        keys = self.key_proj(keys)
        values = self.value_proj(values)

        # Extract some shapes
        num_heads = self.num_heads
        B, L, D = queries.shape

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)

        # Add RoPE to the queries and keys and combine them with the cache
        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # Finally perform the attention computation
        scale = math.sqrt(1 / queries.shape[-1])
        scores = (queries * scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores = scores + mask
        scores = mx.softmax(scores, axis=-1)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

        # Note that we return the keys and values to possibly be used as a cache
        return self.out_proj(values_hat), (keys, values)
```

Copy to clipboard

### Encoder layer[#](#encoder-layer "Link to this heading")

The other component of the Llama model is the encoder layer which uses RMS
normalization [[2]](#id5) and SwiGLU. [[3]](#id6) For RMS normalization we will use
[`mlx.nn.RMSNorm`](../python/nn/_autosummary/mlx.nn.RMSNorm.html#mlx.nn.RMSNorm "mlx.nn.RMSNorm") that is already provided in `mlx.nn`.

```python
class LlamaEncoderLayer(nn.Module):
    def __init__(self, dims: int, mlp_dims: int, num_heads: int):
        super().__init__()

        self.attention = LlamaAttention(dims, num_heads)

        self.norm1 = nn.RMSNorm(dims)
        self.norm2 = nn.RMSNorm(dims)

        self.linear1 = nn.Linear(dims, mlp_dims, bias=False)
        self.linear2 = nn.Linear(dims, mlp_dims, bias=False)
        self.linear3 = nn.Linear(mlp_dims, dims, bias=False)

    def __call__(self, x, mask=None, cache=None):
        y = self.norm1(x)
        y, cache = self.attention(y, y, y, mask, cache)
        x = x + y

        y = self.norm2(x)
        a = self.linear1(y)
        b = self.linear2(y)
        y = a * mx.sigmoid(a) * b
        y = self.linear3(y)
        x = x + y

        return x, cache
```

Copy to clipboard

### Full model[#](#full-model "Link to this heading")

To implement any Llama model we simply have to combine `LlamaEncoderLayer`
instances with an [`mlx.nn.Embedding`](../python/nn/_autosummary/mlx.nn.Embedding.html#mlx.nn.Embedding "mlx.nn.Embedding") to embed the input tokens.

```python
class Llama(nn.Module):
    def __init__(
        self, num_layers: int, vocab_size: int, dims: int, mlp_dims: int, num_heads: int
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dims)
        self.layers = [
            LlamaEncoderLayer(dims, mlp_dims, num_heads) for _ in range(num_layers)
        ]
        self.norm = nn.RMSNorm(dims)
        self.out_proj = nn.Linear(dims, vocab_size, bias=False)

    def __call__(self, x):
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.embedding.weight.dtype)

        x = self.embedding(x)
        for l in self.layers:
            x, _ = l(x, mask)
        x = self.norm(x)
        return self.out_proj(x)
```

Copy to clipboard

Note that in the implementation above we use a simple list to hold the encoder
layers but using `model.parameters()` will still consider these layers.

### Generation[#](#generation "Link to this heading")

Our `Llama` module can be used for training but not inference as the
`__call__` method above processes one input, completely ignores the cache and
performs no sampling whatsoever. In the rest of this subsection, we will
implement the inference function as a python generator that processes the
prompt and then autoregressively yields tokens one at a time.

```python
class Llama(nn.Module):
    ...

    def generate(self, x, temp=1.0):
        cache = []

        # Make an additive causal mask. We will need that to process the prompt.
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.embedding.weight.dtype)

        # First we process the prompt x the same way as in __call__ but
        # save the caches in cache
        x = self.embedding(x)
        for l in self.layers:
            x, c = l(x, mask=mask)
            cache.append(c)  # <--- we store the per layer cache in a
                             #      simple python list
        x = self.norm(x)
        y = self.out_proj(x[:, -1])  # <--- we only care about the last logits
                                     #      that generate the next token
        y = mx.random.categorical(y * (1/temp))

        # y now has size [1]
        # Since MLX is lazily evaluated nothing is computed yet.
        # Calling y.item() would force the computation to happen at
        # this point but we can also choose not to do that and let the
        # user choose when to start the computation.
        yield y

        # Now we parsed the prompt and generated the first token we
        # need to feed it back into the model and loop to generate the
        # rest.
        while True:
            # Unsqueezing the last dimension to add a sequence length
            # dimension of 1
            x = y[:, None]

            x = self.embedding(x)
            for i in range(len(cache)):
                # We are overwriting the arrays in the cache list. When
                # the computation will happen, MLX will be discarding the
                # old cache the moment it is not needed anymore.
                x, cache[i] = self.layers[i](x, mask=None, cache=cache[i])
            x = self.norm(x)
            y = self.out_proj(x[:, -1])
            y = mx.random.categorical(y * (1/temp))

            yield y
```

Copy to clipboard

### Putting it all together[#](#putting-it-all-together "Link to this heading")

We now have everything we need to create a Llama model and sample tokens from
it. In the following code, we randomly initialize a small Llama model, process
6 tokens of prompt and generate 10 tokens.

```python
model = Llama(num_layers=12, vocab_size=8192, dims=512, mlp_dims=1024, num_heads=8)

# Since MLX is lazily evaluated nothing has actually been materialized yet.
# We could have set the `dims` to 20_000 on a machine with 8GB of RAM and the
# code above would still run. Let's actually materialize the model.
mx.eval(model.parameters())

prompt = mx.array([[1, 10, 8, 32, 44, 7]])  # <-- Note the double brackets because we
                                            #     have a batch dimension even
                                            #     though it is 1 in this case

generated = [t for i, t in zip(range(10), model.generate(prompt, 0.8))]

# Since we haven't evaluated anything, nothing is computed yet. The list
# `generated` contains the arrays that hold the computation graph for the
# full processing of the prompt and the generation of 10 tokens.
#
# We can evaluate them one at a time, or all together. Concatenate them or
# print them. They would all result in very similar runtimes and give exactly
# the same results.
mx.eval(generated)
```

Copy to clipboard

## Converting the weights[#](#converting-the-weights "Link to this heading")

This section assumes that you have access to the original Llama weights and the
SentencePiece model that comes with them. We will write a small script to
convert the PyTorch weights to MLX compatible ones and write them in a NPZ file
that can be loaded directly by MLX.

```python
import argparse
from itertools import starmap

import numpy as np
import torch

def map_torch_to_mlx(key, value):
    if "tok_embedding" in key:
        key = "embedding.weight"

    elif "norm" in key:
        key = key.replace("attention_norm", "norm1").replace("ffn_norm", "norm2")

    elif "wq" in key or "wk" in key or "wv" in key or "wo" in key:
        key = key.replace("wq", "query_proj")
        key = key.replace("wk", "key_proj")
        key = key.replace("wv", "value_proj")
        key = key.replace("wo", "out_proj")

    elif "w1" in key or "w2" in key or "w3" in key:
        # The FFN is a separate submodule in PyTorch
        key = key.replace("feed_forward.w1", "linear1")
        key = key.replace("feed_forward.w3", "linear2")
        key = key.replace("feed_forward.w2", "linear3")

    elif "output" in key:
        key = key.replace("output", "out_proj")

    elif "rope" in key:
        return None, None

    return key, value.numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Llama weights to MLX")
    parser.add_argument("torch_weights")
    parser.add_argument("output_file")
    args = parser.parse_args()

    state = torch.load(args.torch_weights)
    np.savez(
        args.output_file,
        **{k: v for k, v in starmap(map_torch_to_mlx, state.items()) if k is not None}
    )
```

Copy to clipboard

## Weight loading and benchmarking[#](#weight-loading-and-benchmarking "Link to this heading")

After converting the weights to be compatible to our implementation, all that is
left is to load them from disk and we can finally use the LLM to generate text.
We can load numpy format files using the [`mlx.core.load()`](../python/_autosummary/mlx.core.load.html#mlx.core.load "mlx.core.load") operation.

To create a parameter dictionary from the key/value representation of NPZ files
we will use the [`mlx.utils.tree_unflatten()`](../python/_autosummary/mlx.utils.tree_unflatten.html#mlx.utils.tree_unflatten "mlx.utils.tree_unflatten") helper method as follows:

```python
from mlx.utils import tree_unflatten

model.update(tree_unflatten(list(mx.load(weight_file).items())))
```

Copy to clipboard

[`mlx.utils.tree_unflatten()`](../python/_autosummary/mlx.utils.tree_unflatten.html#mlx.utils.tree_unflatten "mlx.utils.tree_unflatten") will take keys from the NPZ file that look
like `layers.2.attention.query_proj.weight` and will transform them to

```python
{"layers": [..., ..., {"attention": {"query_proj": {"weight": ...}}}]}
```

Copy to clipboard

which can then be used to update the model. Note that the method above incurs
several unnecessary copies from disk to numpy and then from numpy to MLX. It
will be replaced in the future with direct loading to MLX.

You can download the full example code in [mlx-examples](https://github.com/ml-explore/mlx-examples/tree/main/llms/llama). Assuming, the
existence of `weights.pth` and `tokenizer.model` in the current working
directory we can play around with our inference script as follows (the timings
are representative of an M1 Ultra and the 7B parameter Llama model):

```python
$ python convert.py weights.pth llama-7B.mlx.npz
$ python llama.py llama-7B.mlx.npz tokenizer.model 'Call me Ishmael. Some years ago never mind how long precisely'
[INFO] Loading model from disk: 5.247 s
Press enter to start generation
------
, having little or no money in my purse, and nothing of greater consequence in my mind, I happened to be walking down Gower Street in the afternoon, in the heavy rain, and I saw a few steps off, a man in rags, who sat upon his bundle and looked hard into the wet as if he were going to cry. I watched him attentively for some time, and could not but observe that, though a numerous crowd was hurrying up and down,
------
[INFO] Prompt processing: 0.437 s
[INFO] Full generation: 4.330 s
```

Copy to clipboard

We observe that 4.3 seconds are required to generate 100 tokens and 0.4 seconds
of those are spent processing the prompt. This amounts to a little over **39 ms
per token**.

By running with a much bigger prompt we can see that the per token generation
time as well as the prompt processing time remains almost constant.

```python
$ python llama.py llama-7B.mlx.npz tokenizer.model 'Call me Ishmael. Some years ago never mind how long precisely, having little or no money in my purse, and nothing of greater consequence in my mind, I happened to be walking down Gower Street in the afternoon, in the heavy rain, and I saw a few steps off, a man in rags, who sat upon his bundle and looked hard into the wet as if he were going to cry. I watched him attentively for some time, and could not but observe that, though a numerous crowd was hurrying up and down, nobody took the least notice of him. I stopped at last, at a little distance, as if I had been in doubt, and after looking on a few minutes, walked straight up to him. He slowly raised his eyes, and fixed them upon me for a moment, without speaking, and then resumed his place and posture as before. I stood looking at him for a while, feeling very much pain at heart, and then said to him, “What are you doing there?” Something like a smile passed over his face, as he said slowly, “I am waiting for someone; but it has been three quarters of an hour now, and he has not come.” “What is it you are waiting for?” said I. Still he made no immediate reply, but again put his face down upon his hands, and did not'
[INFO] Loading model from disk: 5.247 s
Press enter to start generation
------
take his eyes from the ground. “What is it you are waiting for?” said I. “I am not accustomed to be thus questioned,” said he. “You look like a reasonable man—tell me, then, what are you waiting for?” “You would not understand,” he replied; “and how could you help me, if I were to tell you?” “I should not only understand, but would do all that I could,” said I. He did not
------
[INFO] Prompt processing: 0.579 s
[INFO] Full generation: 4.690 s
$ python llama.py --num-tokens 500 llama-7B.mlx.npz tokenizer.model 'Call me Ishmael. Some years ago never mind how long precisely, having little or no money in my purse, and nothing of greater consequence in my mind, I happened to be walking down Gower Street in the afternoon, in the heavy rain, and I saw a few steps off, a man in rags, who sat upon his bundle and looked hard into the wet as if he were going to cry. I watched him attentively for some time, and could not but observe that, though a numerous crowd was hurrying up and down, nobody took the least notice of him. I stopped at last, at a little distance, as if I had been in doubt, and after looking on a few minutes, walked straight up to him. He slowly raised his eyes, and fixed them upon me for a moment, without speaking, and then resumed his place and posture as before. I stood looking at him for a while, feeling very much pain at heart, and then said to him, “What are you doing there?” Something like a smile passed over his face, as he said slowly, “I am waiting for someone; but it has been three quarters of an hour now, and he has not come.” “What is it you are waiting for?” said I. Still he made no immediate reply, but again put his face down upon his hands, and did not'
[INFO] Loading model from disk: 5.628 s
Press enter to start generation
------
take his eyes from the ground. “What is it you are waiting for?” said I. “I am not accustomed to be thus questioned,” said he. “You look like a reasonable man—tell me, then, what are you waiting for?” “You would not understand,” he replied; “and how could you help me, if I were to tell you?” “I should not only understand, but would do all that I could,” said I. He did not reply, but still went on looking at the ground, and took hold of his bundle with a nervous trembling. I waited some time, and then resumed. “It is of no use to say you would not understand, if I were to tell you,” said he. “I have not told you why I am waiting for him,” said I. “And I am sure I should not understand,” replied he. “I will tell you then,” said I, “and, perhaps, you would not be surprised.” “No matter,” said he, “I shall be surprised anyhow; so tell me why you are waiting for him.” “He is my friend,” said I. “Yes,” said he, with a slight smile, “I know.” “He has been kind to me,” said I, “and I am waiting for him. I want to see him, and could have waited as I am now, for a much longer time.” “He will not soon come,” said he. “Unless he sees you here, he will not know of your having waited, and he will be very unlikely to come.” “No matter,” said I, “I shall wait for him.” “This is a strange thing,” said he, still with the same amused smile. “How did you know,” said I, “that he was coming? How should you be waiting?” “That is my secret,” said he. “And you expect him?” “Yes,” said I. “Are you disappointed then, if he does not come?” “No,” said I, “it is his secret, not mine.” “If he comes,” said he, “do you mean to go straight away?” “Yes,” said I, “I cannot be happy if I do not go straight away after him.” “Did you know this place before?” asked he. “Yes,” said I. “Is there any shop to buy food here?” “
------
[INFO] Prompt processing: 0.633 s
[INFO] Full generation: 21.475 s
```

Copy to clipboard

## Scripts[#](#scripts "Link to this heading")

Download the code

The full example code is available in [mlx-examples](https://github.com/ml-explore/mlx-examples/tree/main/llms/llama).

[[1](#id1)]

Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B. and Liu, Y., 2021.
Roformer: Enhanced transformer with rotary position embedding. arXiv
preprint arXiv:2104.09864.


[[2](#id2)]

Zhang, B. and Sennrich, R., 2019. Root mean square layer normalization.
Advances in Neural Information Processing Systems, 32.


[[3](#id3)]

Shazeer, N., 2020. Glu variants improve transformer. arXiv preprint
arXiv:2002.05202.

Contents

---

# mlx_ops

Source: https://ml-explore.github.io/mlx/build/html/python/ops.html

---

* [.rst](../_sources/python/ops.rst)
* .pdf

# Operations

# Operations[#](#operations "Link to this heading")

|  |  |
| --- | --- |
| [`abs`](_autosummary/mlx.core.abs.html#mlx.core.abs "mlx.core.abs")(a, /, \*[, stream]) | Element-wise absolute value. |
| [`add`](_autosummary/mlx.core.add.html#mlx.core.add "mlx.core.add")(a, b[, stream]) | Element-wise addition. |
| [`addmm`](_autosummary/mlx.core.addmm.html#mlx.core.addmm "mlx.core.addmm")(c, a, b, /[, alpha, beta, stream]) | Matrix multiplication with addition and optional scaling. |
| [`all`](_autosummary/mlx.core.all.html#mlx.core.all "mlx.core.all")(a, /[, axis, keepdims, stream]) | An and reduction over the given axes. |
| [`allclose`](_autosummary/mlx.core.allclose.html#mlx.core.allclose "mlx.core.allclose")(a, b, /[, rtol, atol, equal\_nan, ...]) | Approximate comparison of two arrays. |
| [`any`](_autosummary/mlx.core.any.html#mlx.core.any "mlx.core.any")(a, /[, axis, keepdims, stream]) | An or reduction over the given axes. |
| [`arange`](_autosummary/mlx.core.arange.html#mlx.core.arange "mlx.core.arange")(-> array) | Overloaded function. |
| [`arccos`](_autosummary/mlx.core.arccos.html#mlx.core.arccos "mlx.core.arccos")(a, /, \*[, stream]) | Element-wise inverse cosine. |
| [`arccosh`](_autosummary/mlx.core.arccosh.html#mlx.core.arccosh "mlx.core.arccosh")(a, /, \*[, stream]) | Element-wise inverse hyperbolic cosine. |
| [`arcsin`](_autosummary/mlx.core.arcsin.html#mlx.core.arcsin "mlx.core.arcsin")(a, /, \*[, stream]) | Element-wise inverse sine. |
| [`arcsinh`](_autosummary/mlx.core.arcsinh.html#mlx.core.arcsinh "mlx.core.arcsinh")(a, /, \*[, stream]) | Element-wise inverse hyperbolic sine. |
| [`arctan`](_autosummary/mlx.core.arctan.html#mlx.core.arctan "mlx.core.arctan")(a, /, \*[, stream]) | Element-wise inverse tangent. |
| [`arctan2`](_autosummary/mlx.core.arctan2.html#mlx.core.arctan2 "mlx.core.arctan2")(a, b, /, \*[, stream]) | Element-wise inverse tangent of the ratio of two arrays. |
| [`arctanh`](_autosummary/mlx.core.arctanh.html#mlx.core.arctanh "mlx.core.arctanh")(a, /, \*[, stream]) | Element-wise inverse hyperbolic tangent. |
| [`argmax`](_autosummary/mlx.core.argmax.html#mlx.core.argmax "mlx.core.argmax")(a, /[, axis, keepdims, stream]) | Indices of the maximum values along the axis. |
| [`argmin`](_autosummary/mlx.core.argmin.html#mlx.core.argmin "mlx.core.argmin")(a, /[, axis, keepdims, stream]) | Indices of the minimum values along the axis. |
| [`argpartition`](_autosummary/mlx.core.argpartition.html#mlx.core.argpartition "mlx.core.argpartition")(a, /, kth[, axis, stream]) | Returns the indices that partition the array. |
| [`argsort`](_autosummary/mlx.core.argsort.html#mlx.core.argsort "mlx.core.argsort")(a, /[, axis, stream]) | Returns the indices that sort the array. |
| [`array_equal`](_autosummary/mlx.core.array_equal.html#mlx.core.array_equal "mlx.core.array_equal")(a, b[, equal\_nan, stream]) | Array equality check. |
| [`as_strided`](_autosummary/mlx.core.as_strided.html#mlx.core.as_strided "mlx.core.as_strided")(a, /[, shape, strides, offset, ...]) | Create a view into the array with the given shape and strides. |
| [`atleast_1d`](_autosummary/mlx.core.atleast_1d.html#mlx.core.atleast_1d "mlx.core.atleast_1d")(\*arys[, stream]) | Convert all arrays to have at least one dimension. |
| [`atleast_2d`](_autosummary/mlx.core.atleast_2d.html#mlx.core.atleast_2d "mlx.core.atleast_2d")(\*arys[, stream]) | Convert all arrays to have at least two dimensions. |
| [`atleast_3d`](_autosummary/mlx.core.atleast_3d.html#mlx.core.atleast_3d "mlx.core.atleast_3d")(\*arys[, stream]) | Convert all arrays to have at least three dimensions. |
| [`bitwise_and`](_autosummary/mlx.core.bitwise_and.html#mlx.core.bitwise_and "mlx.core.bitwise_and")(a, b[, stream]) | Element-wise bitwise and. |
| [`bitwise_invert`](_autosummary/mlx.core.bitwise_invert.html#mlx.core.bitwise_invert "mlx.core.bitwise_invert")(a[, stream]) | Element-wise bitwise inverse. |
| [`bitwise_or`](_autosummary/mlx.core.bitwise_or.html#mlx.core.bitwise_or "mlx.core.bitwise_or")(a, b[, stream]) | Element-wise bitwise or. |
| [`bitwise_xor`](_autosummary/mlx.core.bitwise_xor.html#mlx.core.bitwise_xor "mlx.core.bitwise_xor")(a, b[, stream]) | Element-wise bitwise xor. |
| [`block_masked_mm`](_autosummary/mlx.core.block_masked_mm.html#mlx.core.block_masked_mm "mlx.core.block_masked_mm")(a, b, /[, block\_size, ...]) | Matrix multiplication with block masking. |
| [`broadcast_arrays`](_autosummary/mlx.core.broadcast_arrays.html#mlx.core.broadcast_arrays "mlx.core.broadcast_arrays")(\*arrays[, stream]) | Broadcast arrays against one another. |
| [`broadcast_to`](_autosummary/mlx.core.broadcast_to.html#mlx.core.broadcast_to "mlx.core.broadcast_to")(a, /, shape, \*[, stream]) | Broadcast an array to the given shape. |
| [`ceil`](_autosummary/mlx.core.ceil.html#mlx.core.ceil "mlx.core.ceil")(a, /, \*[, stream]) | Element-wise ceil. |
| [`clip`](_autosummary/mlx.core.clip.html#mlx.core.clip "mlx.core.clip")(a, /, a\_min, a\_max, \*[, stream]) | Clip the values of the array between the given minimum and maximum. |
| [`concatenate`](_autosummary/mlx.core.concatenate.html#mlx.core.concatenate "mlx.core.concatenate")(arrays[, axis, stream]) | Concatenate the arrays along the given axis. |
| [`contiguous`](_autosummary/mlx.core.contiguous.html#mlx.core.contiguous "mlx.core.contiguous")(a, /[, allow\_col\_major, stream]) | Force an array to be row contiguous. |
| [`conj`](_autosummary/mlx.core.conj.html#mlx.core.conj "mlx.core.conj")(a, \*[, stream]) | Return the elementwise complex conjugate of the input. |
| [`conjugate`](_autosummary/mlx.core.conjugate.html#mlx.core.conjugate "mlx.core.conjugate")(a, \*[, stream]) | Return the elementwise complex conjugate of the input. |
| [`convolve`](_autosummary/mlx.core.convolve.html#mlx.core.convolve "mlx.core.convolve")(a, v, /[, mode, stream]) | The discrete convolution of 1D arrays. |
| [`conv1d`](_autosummary/mlx.core.conv1d.html#mlx.core.conv1d "mlx.core.conv1d")(input, weight, /[, stride, padding, ...]) | 1D convolution over an input with several channels |
| [`conv2d`](_autosummary/mlx.core.conv2d.html#mlx.core.conv2d "mlx.core.conv2d")(input, weight, /[, stride, padding, ...]) | 2D convolution over an input with several channels |
| [`conv3d`](_autosummary/mlx.core.conv3d.html#mlx.core.conv3d "mlx.core.conv3d")(input, weight, /[, stride, padding, ...]) | 3D convolution over an input with several channels |
| [`conv_transpose1d`](_autosummary/mlx.core.conv_transpose1d.html#mlx.core.conv_transpose1d "mlx.core.conv_transpose1d")(input, weight, /[, stride, ...]) | 1D transposed convolution over an input with several channels |
| [`conv_transpose2d`](_autosummary/mlx.core.conv_transpose2d.html#mlx.core.conv_transpose2d "mlx.core.conv_transpose2d")(input, weight, /[, stride, ...]) | 2D transposed convolution over an input with several channels |
| [`conv_transpose3d`](_autosummary/mlx.core.conv_transpose3d.html#mlx.core.conv_transpose3d "mlx.core.conv_transpose3d")(input, weight, /[, stride, ...]) | 3D transposed convolution over an input with several channels |
| [`conv_general`](_autosummary/mlx.core.conv_general.html#mlx.core.conv_general "mlx.core.conv_general")(input, weight, /[, stride, ...]) | General convolution over an input with several channels |
| [`cos`](_autosummary/mlx.core.cos.html#mlx.core.cos "mlx.core.cos")(a, /, \*[, stream]) | Element-wise cosine. |
| [`cosh`](_autosummary/mlx.core.cosh.html#mlx.core.cosh "mlx.core.cosh")(a, /, \*[, stream]) | Element-wise hyperbolic cosine. |
| [`cummax`](_autosummary/mlx.core.cummax.html#mlx.core.cummax "mlx.core.cummax")(a, /[, axis, reverse, inclusive, stream]) | Return the cumulative maximum of the elements along the given axis. |
| [`cummin`](_autosummary/mlx.core.cummin.html#mlx.core.cummin "mlx.core.cummin")(a, /[, axis, reverse, inclusive, stream]) | Return the cumulative minimum of the elements along the given axis. |
| [`cumprod`](_autosummary/mlx.core.cumprod.html#mlx.core.cumprod "mlx.core.cumprod")(a, /[, axis, reverse, inclusive, stream]) | Return the cumulative product of the elements along the given axis. |
| [`cumsum`](_autosummary/mlx.core.cumsum.html#mlx.core.cumsum "mlx.core.cumsum")(a, /[, axis, reverse, inclusive, stream]) | Return the cumulative sum of the elements along the given axis. |
| [`degrees`](_autosummary/mlx.core.degrees.html#mlx.core.degrees "mlx.core.degrees")(a, /, \*[, stream]) | Convert angles from radians to degrees. |
| [`dequantize`](_autosummary/mlx.core.dequantize.html#mlx.core.dequantize "mlx.core.dequantize")(w, /, scales[, biases, ...]) | Dequantize the matrix `w` using quantization parameters. |
| [`diag`](_autosummary/mlx.core.diag.html#mlx.core.diag "mlx.core.diag")(a, /[, k, stream]) | Extract a diagonal or construct a diagonal matrix. |
| [`diagonal`](_autosummary/mlx.core.diagonal.html#mlx.core.diagonal "mlx.core.diagonal")(a[, offset, axis1, axis2, stream]) | Return specified diagonals. |
| [`divide`](_autosummary/mlx.core.divide.html#mlx.core.divide "mlx.core.divide")(a, b[, stream]) | Element-wise division. |
| [`divmod`](_autosummary/mlx.core.divmod.html#mlx.core.divmod "mlx.core.divmod")(a, b[, stream]) | Element-wise quotient and remainder. |
| [`einsum`](_autosummary/mlx.core.einsum.html#mlx.core.einsum "mlx.core.einsum")(subscripts, \*operands[, stream]) | Perform the Einstein summation convention on the operands. |
| [`einsum_path`](_autosummary/mlx.core.einsum_path.html#mlx.core.einsum_path "mlx.core.einsum_path")(subscripts, \*operands) | Compute the contraction order for the given Einstein summation. |
| [`equal`](_autosummary/mlx.core.equal.html#mlx.core.equal "mlx.core.equal")(a, b[, stream]) | Element-wise equality. |
| [`erf`](_autosummary/mlx.core.erf.html#mlx.core.erf "mlx.core.erf")(a, /, \*[, stream]) | Element-wise error function. |
| [`erfinv`](_autosummary/mlx.core.erfinv.html#mlx.core.erfinv "mlx.core.erfinv")(a, /, \*[, stream]) | Element-wise inverse of [`erf()`](_autosummary/mlx.core.erf.html#mlx.core.erf "mlx.core.erf"). |
| [`exp`](_autosummary/mlx.core.exp.html#mlx.core.exp "mlx.core.exp")(a, /, \*[, stream]) | Element-wise exponential. |
| [`expm1`](_autosummary/mlx.core.expm1.html#mlx.core.expm1 "mlx.core.expm1")(a, /, \*[, stream]) | Element-wise exponential minus 1. |
| [`expand_dims`](_autosummary/mlx.core.expand_dims.html#mlx.core.expand_dims "mlx.core.expand_dims")(a, /, axis, \*[, stream]) | Add a size one dimension at the given axis. |
| [`eye`](_autosummary/mlx.core.eye.html#mlx.core.eye "mlx.core.eye")(n[, m, k, dtype, stream]) | Create an identity matrix or a general diagonal matrix. |
| [`flatten`](_autosummary/mlx.core.flatten.html#mlx.core.flatten "mlx.core.flatten")(a, /[, start\_axis, end\_axis, stream]) | Flatten an array. |
| [`floor`](_autosummary/mlx.core.floor.html#mlx.core.floor "mlx.core.floor")(a, /, \*[, stream]) | Element-wise floor. |
| [`floor_divide`](_autosummary/mlx.core.floor_divide.html#mlx.core.floor_divide "mlx.core.floor_divide")(a, b[, stream]) | Element-wise integer division. |
| [`full`](_autosummary/mlx.core.full.html#mlx.core.full "mlx.core.full")(shape, vals[, dtype, stream]) | Construct an array with the given value. |
| [`gather_mm`](_autosummary/mlx.core.gather_mm.html#mlx.core.gather_mm "mlx.core.gather_mm")(a, b, /, lhs\_indices, rhs\_indices, \*) | Matrix multiplication with matrix-level gather. |
| [`gather_qmm`](_autosummary/mlx.core.gather_qmm.html#mlx.core.gather_qmm "mlx.core.gather_qmm")(x, w, /, scales[, biases, ...]) | Perform quantized matrix multiplication with matrix-level gather. |
| [`greater`](_autosummary/mlx.core.greater.html#mlx.core.greater "mlx.core.greater")(a, b[, stream]) | Element-wise greater than. |
| [`greater_equal`](_autosummary/mlx.core.greater_equal.html#mlx.core.greater_equal "mlx.core.greater_equal")(a, b[, stream]) | Element-wise greater or equal. |
| [`hadamard_transform`](_autosummary/mlx.core.hadamard_transform.html#mlx.core.hadamard_transform "mlx.core.hadamard_transform")(a[, scale, stream]) | Perform the Walsh-Hadamard transform along the final axis. |
| [`identity`](_autosummary/mlx.core.identity.html#mlx.core.identity "mlx.core.identity")(n[, dtype, stream]) | Create a square identity matrix. |
| [`imag`](_autosummary/mlx.core.imag.html#mlx.core.imag "mlx.core.imag")(a, /, \*[, stream]) | Returns the imaginary part of a complex array. |
| [`inner`](_autosummary/mlx.core.inner.html#mlx.core.inner "mlx.core.inner")(a, b, /, \*[, stream]) | Ordinary inner product of vectors for 1-D arrays, in higher dimensions a sum product over the last axes. |
| [`isfinite`](_autosummary/mlx.core.isfinite.html#mlx.core.isfinite "mlx.core.isfinite")(a[, stream]) | Return a boolean array indicating which elements are finite. |
| [`isclose`](_autosummary/mlx.core.isclose.html#mlx.core.isclose "mlx.core.isclose")(a, b, /[, rtol, atol, equal\_nan, stream]) | Returns a boolean array where two arrays are element-wise equal within a tolerance. |
| [`isinf`](_autosummary/mlx.core.isinf.html#mlx.core.isinf "mlx.core.isinf")(a[, stream]) | Return a boolean array indicating which elements are +/- inifnity. |
| [`isnan`](_autosummary/mlx.core.isnan.html#mlx.core.isnan "mlx.core.isnan")(a[, stream]) | Return a boolean array indicating which elements are NaN. |
| [`isneginf`](_autosummary/mlx.core.isneginf.html#mlx.core.isneginf "mlx.core.isneginf")(a[, stream]) | Return a boolean array indicating which elements are negative infinity. |
| [`isposinf`](_autosummary/mlx.core.isposinf.html#mlx.core.isposinf "mlx.core.isposinf")(a[, stream]) | Return a boolean array indicating which elements are positive infinity. |
| [`issubdtype`](_autosummary/mlx.core.issubdtype.html#mlx.core.issubdtype "mlx.core.issubdtype")(arg1, arg2) | Check if a [`Dtype`](_autosummary/mlx.core.Dtype.html#mlx.core.Dtype "mlx.core.Dtype") or [`DtypeCategory`](_autosummary/mlx.core.DtypeCategory.html#mlx.core.DtypeCategory "mlx.core.DtypeCategory") is a subtype of another. |
| [`kron`](_autosummary/mlx.core.kron.html#mlx.core.kron "mlx.core.kron")(a, b, \*[, stream]) | Compute the Kronecker product of two arrays `a` and `b`. |
| [`left_shift`](_autosummary/mlx.core.left_shift.html#mlx.core.left_shift "mlx.core.left_shift")(a, b[, stream]) | Element-wise left shift. |
| [`less`](_autosummary/mlx.core.less.html#mlx.core.less "mlx.core.less")(a, b[, stream]) | Element-wise less than. |
| [`less_equal`](_autosummary/mlx.core.less_equal.html#mlx.core.less_equal "mlx.core.less_equal")(a, b[, stream]) | Element-wise less than or equal. |
| [`linspace`](_autosummary/mlx.core.linspace.html#mlx.core.linspace "mlx.core.linspace")(start, stop[, num, dtype, stream]) | Generate `num` evenly spaced numbers over interval `[start, stop]`. |
| [`load`](_autosummary/mlx.core.load.html#mlx.core.load "mlx.core.load")(file, /[, format, return\_metadata, stream]) | Load array(s) from a binary file. |
| [`log`](_autosummary/mlx.core.log.html#mlx.core.log "mlx.core.log")(a, /, \*[, stream]) | Element-wise natural logarithm. |
| [`log2`](_autosummary/mlx.core.log2.html#mlx.core.log2 "mlx.core.log2")(a, /, \*[, stream]) | Element-wise base-2 logarithm. |
| [`log10`](_autosummary/mlx.core.log10.html#mlx.core.log10 "mlx.core.log10")(a, /, \*[, stream]) | Element-wise base-10 logarithm. |
| [`log1p`](_autosummary/mlx.core.log1p.html#mlx.core.log1p "mlx.core.log1p")(a, /, \*[, stream]) | Element-wise natural log of one plus the array. |
| [`logaddexp`](_autosummary/mlx.core.logaddexp.html#mlx.core.logaddexp "mlx.core.logaddexp")(a, b, /, \*[, stream]) | Element-wise log-add-exp. |
| [`logcumsumexp`](_autosummary/mlx.core.logcumsumexp.html#mlx.core.logcumsumexp "mlx.core.logcumsumexp")(a, /[, axis, reverse, ...]) | Return the cumulative logsumexp of the elements along the given axis. |
| [`logical_not`](_autosummary/mlx.core.logical_not.html#mlx.core.logical_not "mlx.core.logical_not")(a, /, \*[, stream]) | Element-wise logical not. |
| [`logical_and`](_autosummary/mlx.core.logical_and.html#mlx.core.logical_and "mlx.core.logical_and")(a, b, /, \*[, stream]) | Element-wise logical and. |
| [`logical_or`](_autosummary/mlx.core.logical_or.html#mlx.core.logical_or "mlx.core.logical_or")(a, b, /, \*[, stream]) | Element-wise logical or. |
| [`logsumexp`](_autosummary/mlx.core.logsumexp.html#mlx.core.logsumexp "mlx.core.logsumexp")(a, /[, axis, keepdims, stream]) | A log-sum-exp reduction over the given axes. |
| [`matmul`](_autosummary/mlx.core.matmul.html#mlx.core.matmul "mlx.core.matmul")(a, b, /, \*[, stream]) | Matrix multiplication. |
| [`max`](_autosummary/mlx.core.max.html#mlx.core.max "mlx.core.max")(a, /[, axis, keepdims, stream]) | A max reduction over the given axes. |
| [`maximum`](_autosummary/mlx.core.maximum.html#mlx.core.maximum "mlx.core.maximum")(a, b, /, \*[, stream]) | Element-wise maximum. |
| [`mean`](_autosummary/mlx.core.mean.html#mlx.core.mean "mlx.core.mean")(a, /[, axis, keepdims, stream]) | Compute the mean(s) over the given axes. |
| [`median`](_autosummary/mlx.core.median.html#mlx.core.median "mlx.core.median")(a, /[, axis, keepdims, stream]) | Compute the median(s) over the given axes. |
| [`meshgrid`](_autosummary/mlx.core.meshgrid.html#mlx.core.meshgrid "mlx.core.meshgrid")(\*arrays[, sparse, indexing, stream]) | Generate multidimensional coordinate grids from 1-D coordinate arrays |
| [`min`](_autosummary/mlx.core.min.html#mlx.core.min "mlx.core.min")(a, /[, axis, keepdims, stream]) | A min reduction over the given axes. |
| [`minimum`](_autosummary/mlx.core.minimum.html#mlx.core.minimum "mlx.core.minimum")(a, b, /, \*[, stream]) | Element-wise minimum. |
| [`moveaxis`](_autosummary/mlx.core.moveaxis.html#mlx.core.moveaxis "mlx.core.moveaxis")(a, /, source, destination, \*[, stream]) | Move an axis to a new position. |
| [`multiply`](_autosummary/mlx.core.multiply.html#mlx.core.multiply "mlx.core.multiply")(a, b[, stream]) | Element-wise multiplication. |
| [`nan_to_num`](_autosummary/mlx.core.nan_to_num.html#mlx.core.nan_to_num "mlx.core.nan_to_num")(a[, nan, posinf, neginf, stream]) | Replace NaN and Inf values with finite numbers. |
| [`negative`](_autosummary/mlx.core.negative.html#mlx.core.negative "mlx.core.negative")(a, /, \*[, stream]) | Element-wise negation. |
| [`not_equal`](_autosummary/mlx.core.not_equal.html#mlx.core.not_equal "mlx.core.not_equal")(a, b[, stream]) | Element-wise not equal. |
| [`ones`](_autosummary/mlx.core.ones.html#mlx.core.ones "mlx.core.ones")(shape[, dtype, stream]) | Construct an array of ones. |
| [`ones_like`](_autosummary/mlx.core.ones_like.html#mlx.core.ones_like "mlx.core.ones_like")(a, /, \*[, stream]) | An array of ones like the input. |
| [`outer`](_autosummary/mlx.core.outer.html#mlx.core.outer "mlx.core.outer")(a, b, /, \*[, stream]) | Compute the outer product of two 1-D arrays, if the array's passed are not 1-D a flatten op will be run beforehand. |
| [`partition`](_autosummary/mlx.core.partition.html#mlx.core.partition "mlx.core.partition")(a, /, kth[, axis, stream]) | Returns a partitioned copy of the array such that the smaller `kth` elements are first. |
| [`pad`](_autosummary/mlx.core.pad.html#mlx.core.pad "mlx.core.pad")(a, pad\_width[, mode, constant\_values, ...]) | Pad an array with a constant value |
| [`power`](_autosummary/mlx.core.power.html#mlx.core.power "mlx.core.power")(a, b, /, \*[, stream]) | Element-wise power operation. |
| [`prod`](_autosummary/mlx.core.prod.html#mlx.core.prod "mlx.core.prod")(a, /[, axis, keepdims, stream]) | An product reduction over the given axes. |
| [`put_along_axis`](_autosummary/mlx.core.put_along_axis.html#mlx.core.put_along_axis "mlx.core.put_along_axis")(a, /, indices, values[, ...]) | Put values along an axis at the specified indices. |
| [`quantize`](_autosummary/mlx.core.quantize.html#mlx.core.quantize "mlx.core.quantize")(w, /[, group\_size, bits, mode, stream]) | Quantize the array `w`. |
| [`quantized_matmul`](_autosummary/mlx.core.quantized_matmul.html#mlx.core.quantized_matmul "mlx.core.quantized_matmul")(x, w, /, scales[, biases, ...]) | Perform the matrix multiplication with the quantized matrix `w`. |
| [`radians`](_autosummary/mlx.core.radians.html#mlx.core.radians "mlx.core.radians")(a, /, \*[, stream]) | Convert angles from degrees to radians. |
| [`real`](_autosummary/mlx.core.real.html#mlx.core.real "mlx.core.real")(a, /, \*[, stream]) | Returns the real part of a complex array. |
| [`reciprocal`](_autosummary/mlx.core.reciprocal.html#mlx.core.reciprocal "mlx.core.reciprocal")(a, /, \*[, stream]) | Element-wise reciprocal. |
| [`remainder`](_autosummary/mlx.core.remainder.html#mlx.core.remainder "mlx.core.remainder")(a, b[, stream]) | Element-wise remainder of division. |
| [`repeat`](_autosummary/mlx.core.repeat.html#mlx.core.repeat "mlx.core.repeat")(array, repeats[, axis, stream]) | Repeat an array along a specified axis. |
| [`reshape`](_autosummary/mlx.core.reshape.html#mlx.core.reshape "mlx.core.reshape")(a, /, shape, \*[, stream]) | Reshape an array while preserving the size. |
| [`right_shift`](_autosummary/mlx.core.right_shift.html#mlx.core.right_shift "mlx.core.right_shift")(a, b[, stream]) | Element-wise right shift. |
| [`roll`](_autosummary/mlx.core.roll.html#mlx.core.roll "mlx.core.roll")(a, shift[, axis, stream]) | Roll array elements along a given axis. |
| [`round`](_autosummary/mlx.core.round.html#mlx.core.round "mlx.core.round")(a, /[, decimals, stream]) | Round to the given number of decimals. |
| [`rsqrt`](_autosummary/mlx.core.rsqrt.html#mlx.core.rsqrt "mlx.core.rsqrt")(a, /, \*[, stream]) | Element-wise reciprocal and square root. |
| [`save`](_autosummary/mlx.core.save.html#mlx.core.save "mlx.core.save")(file, arr) | Save the array to a binary file in `.npy` format. |
| [`savez`](_autosummary/mlx.core.savez.html#mlx.core.savez "mlx.core.savez")(file, \*args, \*\*kwargs) | Save several arrays to a binary file in uncompressed `.npz` format. |
| [`savez_compressed`](_autosummary/mlx.core.savez_compressed.html#mlx.core.savez_compressed "mlx.core.savez_compressed")(file, \*args, \*\*kwargs) | Save several arrays to a binary file in compressed `.npz` format. |
| [`save_gguf`](_autosummary/mlx.core.save_gguf.html#mlx.core.save_gguf "mlx.core.save_gguf")(file, arrays, metadata) | Save array(s) to a binary file in `.gguf` format. |
| [`save_safetensors`](_autosummary/mlx.core.save_safetensors.html#mlx.core.save_safetensors "mlx.core.save_safetensors")(file, arrays[, metadata]) | Save array(s) to a binary file in `.safetensors` format. |
| [`sigmoid`](_autosummary/mlx.core.sigmoid.html#mlx.core.sigmoid "mlx.core.sigmoid")(a, /, \*[, stream]) | Element-wise logistic sigmoid. |
| [`sign`](_autosummary/mlx.core.sign.html#mlx.core.sign "mlx.core.sign")(a, /, \*[, stream]) | Element-wise sign. |
| [`sin`](_autosummary/mlx.core.sin.html#mlx.core.sin "mlx.core.sin")(a, /, \*[, stream]) | Element-wise sine. |
| [`sinh`](_autosummary/mlx.core.sinh.html#mlx.core.sinh "mlx.core.sinh")(a, /, \*[, stream]) | Element-wise hyperbolic sine. |
| [`slice`](_autosummary/mlx.core.slice.html#mlx.core.slice "mlx.core.slice")(a, start\_indices, axes, slice\_size, \*) | Extract a sub-array from the input array. |
| [`slice_update`](_autosummary/mlx.core.slice_update.html#mlx.core.slice_update "mlx.core.slice_update")(a, update, start\_indices, axes, \*) | Update a sub-array of the input array. |
| [`softmax`](_autosummary/mlx.core.softmax.html#mlx.core.softmax "mlx.core.softmax")(a, /[, axis, stream]) | Perform the softmax along the given axis. |
| [`sort`](_autosummary/mlx.core.sort.html#mlx.core.sort "mlx.core.sort")(a, /[, axis, stream]) | Returns a sorted copy of the array. |
| [`split`](_autosummary/mlx.core.split.html#mlx.core.split "mlx.core.split")(a, /, indices\_or\_sections[, axis, stream]) | Split an array along a given axis. |
| [`sqrt`](_autosummary/mlx.core.sqrt.html#mlx.core.sqrt "mlx.core.sqrt")(a, /, \*[, stream]) | Element-wise square root. |
| [`square`](_autosummary/mlx.core.square.html#mlx.core.square "mlx.core.square")(a, /, \*[, stream]) | Element-wise square. |
| [`squeeze`](_autosummary/mlx.core.squeeze.html#mlx.core.squeeze "mlx.core.squeeze")(a, /[, axis, stream]) | Remove length one axes from an array. |
| [`stack`](_autosummary/mlx.core.stack.html#mlx.core.stack "mlx.core.stack")(arrays[, axis, stream]) | Stacks the arrays along a new axis. |
| [`std`](_autosummary/mlx.core.std.html#mlx.core.std "mlx.core.std")(a, /[, axis, keepdims, ddof, stream]) | Compute the standard deviation(s) over the given axes. |
| [`stop_gradient`](_autosummary/mlx.core.stop_gradient.html#mlx.core.stop_gradient "mlx.core.stop_gradient")(a, /, \*[, stream]) | Stop gradients from being computed. |
| [`subtract`](_autosummary/mlx.core.subtract.html#mlx.core.subtract "mlx.core.subtract")(a, b[, stream]) | Element-wise subtraction. |
| [`sum`](_autosummary/mlx.core.sum.html#mlx.core.sum "mlx.core.sum")(a, /[, axis, keepdims, stream]) | Sum reduce the array over the given axes. |
| [`swapaxes`](_autosummary/mlx.core.swapaxes.html#mlx.core.swapaxes "mlx.core.swapaxes")(a, /, axis1, axis2, \*[, stream]) | Swap two axes of an array. |
| [`take`](_autosummary/mlx.core.take.html#mlx.core.take "mlx.core.take")(a, /, indices[, axis, stream]) | Take elements along an axis. |
| [`take_along_axis`](_autosummary/mlx.core.take_along_axis.html#mlx.core.take_along_axis "mlx.core.take_along_axis")(a, /, indices[, axis, stream]) | Take values along an axis at the specified indices. |
| [`tan`](_autosummary/mlx.core.tan.html#mlx.core.tan "mlx.core.tan")(a, /, \*[, stream]) | Element-wise tangent. |
| [`tanh`](_autosummary/mlx.core.tanh.html#mlx.core.tanh "mlx.core.tanh")(a, /, \*[, stream]) | Element-wise hyperbolic tangent. |
| [`tensordot`](_autosummary/mlx.core.tensordot.html#mlx.core.tensordot "mlx.core.tensordot")(a, b, /[, axes, stream]) | Compute the tensor dot product along the specified axes. |
| [`tile`](_autosummary/mlx.core.tile.html#mlx.core.tile "mlx.core.tile")(a, reps, /, \*[, stream]) | Construct an array by repeating `a` the number of times given by `reps`. |
| [`topk`](_autosummary/mlx.core.topk.html#mlx.core.topk "mlx.core.topk")(a, /, k[, axis, stream]) | Returns the `k` largest elements from the input along a given axis. |
| [`trace`](_autosummary/mlx.core.trace.html#mlx.core.trace "mlx.core.trace")(a, /[, offset, axis1, axis2, dtype, ...]) | Return the sum along a specified diagonal in the given array. |
| [`transpose`](_autosummary/mlx.core.transpose.html#mlx.core.transpose "mlx.core.transpose")(a, /[, axes, stream]) | Transpose the dimensions of the array. |
| [`tri`](_autosummary/mlx.core.tri.html#mlx.core.tri "mlx.core.tri")(n, m, k[, dtype, stream]) | An array with ones at and below the given diagonal and zeros elsewhere. |
| [`tril`](_autosummary/mlx.core.tril.html#mlx.core.tril "mlx.core.tril")(x, k, \*[, stream]) | Zeros the array above the given diagonal. |
| [`triu`](_autosummary/mlx.core.triu.html#mlx.core.triu "mlx.core.triu")(x, k, \*[, stream]) | Zeros the array below the given diagonal. |
| [`unflatten`](_autosummary/mlx.core.unflatten.html#mlx.core.unflatten "mlx.core.unflatten")(a, /, axis, shape, \*[, stream]) | Unflatten an axis of an array to a shape. |
| [`var`](_autosummary/mlx.core.var.html#mlx.core.var "mlx.core.var")(a, /[, axis, keepdims, ddof, stream]) | Compute the variance(s) over the given axes. |
| [`view`](_autosummary/mlx.core.view.html#mlx.core.view "mlx.core.view")(a, dtype[, stream]) | View the array as a different type. |
| [`where`](_autosummary/mlx.core.where.html#mlx.core.where "mlx.core.where")(condition, x, y, /, \*[, stream]) | Select from `x` or `y` according to `condition`. |
| [`zeros`](_autosummary/mlx.core.zeros.html#mlx.core.zeros "mlx.core.zeros")(shape[, dtype, stream]) | Construct an array of zeros. |
| [`zeros_like`](_autosummary/mlx.core.zeros_like.html#mlx.core.zeros_like "mlx.core.zeros_like")(a, /, \*[, stream]) | An array of zeros like the input. |

---

# mlx_array

Source: https://ml-explore.github.io/mlx/build/html/python/array.html

---

* [.rst](../_sources/python/array.rst)
* .pdf

# Array

# Array[#](#array "Link to this heading")

|  |  |
| --- | --- |
| [`array`](_autosummary/mlx.core.array.html#mlx.core.array "mlx.core.array") | An N-dimensional array object. |
| [`array.astype`](_autosummary/mlx.core.array.astype.html#mlx.core.array.astype "mlx.core.array.astype")(self, dtype[, stream]) | Cast the array to a specified type. |
| [`array.at`](_autosummary/mlx.core.array.at.html#mlx.core.array.at "mlx.core.array.at") | Used to apply updates at the given indices. |
| [`array.item`](_autosummary/mlx.core.array.item.html#mlx.core.array.item "mlx.core.array.item")(self) | Access the value of a scalar array. |
| [`array.tolist`](_autosummary/mlx.core.array.tolist.html#mlx.core.array.tolist "mlx.core.array.tolist")(self) | Convert the array to a Python [`list`](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)"). |
| [`array.dtype`](_autosummary/mlx.core.array.dtype.html#mlx.core.array.dtype "mlx.core.array.dtype") | The array's [`Dtype`](_autosummary/mlx.core.Dtype.html#mlx.core.Dtype "mlx.core.Dtype"). |
| [`array.itemsize`](_autosummary/mlx.core.array.itemsize.html#mlx.core.array.itemsize "mlx.core.array.itemsize") | The size of the array's datatype in bytes. |
| [`array.nbytes`](_autosummary/mlx.core.array.nbytes.html#mlx.core.array.nbytes "mlx.core.array.nbytes") | The number of bytes in the array. |
| [`array.ndim`](_autosummary/mlx.core.array.ndim.html#mlx.core.array.ndim "mlx.core.array.ndim") | The array's dimension. |
| [`array.shape`](_autosummary/mlx.core.array.shape.html#mlx.core.array.shape "mlx.core.array.shape") | The shape of the array as a Python tuple. |
| [`array.size`](_autosummary/mlx.core.array.size.html#mlx.core.array.size "mlx.core.array.size") | Number of elements in the array. |
| [`array.real`](_autosummary/mlx.core.array.real.html#mlx.core.array.real "mlx.core.array.real") | The real part of a complex array. |
| [`array.imag`](_autosummary/mlx.core.array.imag.html#mlx.core.array.imag "mlx.core.array.imag") | The imaginary part of a complex array. |
| [`array.abs`](_autosummary/mlx.core.array.abs.html#mlx.core.array.abs "mlx.core.array.abs")(self, \*[, stream]) | See [`abs()`](_autosummary/mlx.core.abs.html#mlx.core.abs "mlx.core.abs"). |
| [`array.all`](_autosummary/mlx.core.array.all.html#mlx.core.array.all "mlx.core.array.all")(self[, axis, keepdims, stream]) | See [`all()`](_autosummary/mlx.core.all.html#mlx.core.all "mlx.core.all"). |
| [`array.any`](_autosummary/mlx.core.array.any.html#mlx.core.array.any "mlx.core.array.any")(self[, axis, keepdims, stream]) | See [`any()`](_autosummary/mlx.core.any.html#mlx.core.any "mlx.core.any"). |
| [`array.argmax`](_autosummary/mlx.core.array.argmax.html#mlx.core.array.argmax "mlx.core.array.argmax")(self[, axis, keepdims, stream]) | See [`argmax()`](_autosummary/mlx.core.argmax.html#mlx.core.argmax "mlx.core.argmax"). |
| [`array.argmin`](_autosummary/mlx.core.array.argmin.html#mlx.core.array.argmin "mlx.core.array.argmin")(self[, axis, keepdims, stream]) | See [`argmin()`](_autosummary/mlx.core.argmin.html#mlx.core.argmin "mlx.core.argmin"). |
| [`array.conj`](_autosummary/mlx.core.array.conj.html#mlx.core.array.conj "mlx.core.array.conj")(self, \*[, stream]) | See [`conj()`](_autosummary/mlx.core.conj.html#mlx.core.conj "mlx.core.conj"). |
| [`array.cos`](_autosummary/mlx.core.array.cos.html#mlx.core.array.cos "mlx.core.array.cos")(self, \*[, stream]) | See [`cos()`](_autosummary/mlx.core.cos.html#mlx.core.cos "mlx.core.cos"). |
| [`array.cummax`](_autosummary/mlx.core.array.cummax.html#mlx.core.array.cummax "mlx.core.array.cummax")(self[, axis, reverse, ...]) | See [`cummax()`](_autosummary/mlx.core.cummax.html#mlx.core.cummax "mlx.core.cummax"). |
| [`array.cummin`](_autosummary/mlx.core.array.cummin.html#mlx.core.array.cummin "mlx.core.array.cummin")(self[, axis, reverse, ...]) | See [`cummin()`](_autosummary/mlx.core.cummin.html#mlx.core.cummin "mlx.core.cummin"). |
| [`array.cumprod`](_autosummary/mlx.core.array.cumprod.html#mlx.core.array.cumprod "mlx.core.array.cumprod")(self[, axis, reverse, ...]) | See [`cumprod()`](_autosummary/mlx.core.cumprod.html#mlx.core.cumprod "mlx.core.cumprod"). |
| [`array.cumsum`](_autosummary/mlx.core.array.cumsum.html#mlx.core.array.cumsum "mlx.core.array.cumsum")(self[, axis, reverse, ...]) | See [`cumsum()`](_autosummary/mlx.core.cumsum.html#mlx.core.cumsum "mlx.core.cumsum"). |
| [`array.diag`](_autosummary/mlx.core.array.diag.html#mlx.core.array.diag "mlx.core.array.diag")(self[, k, stream]) | Extract a diagonal or construct a diagonal matrix. |
| [`array.diagonal`](_autosummary/mlx.core.array.diagonal.html#mlx.core.array.diagonal "mlx.core.array.diagonal")(self[, offset, axis1, axis2, ...]) | See [`diagonal()`](_autosummary/mlx.core.diagonal.html#mlx.core.diagonal "mlx.core.diagonal"). |
| [`array.exp`](_autosummary/mlx.core.array.exp.html#mlx.core.array.exp "mlx.core.array.exp")(self, \*[, stream]) | See [`exp()`](_autosummary/mlx.core.exp.html#mlx.core.exp "mlx.core.exp"). |
| [`array.flatten`](_autosummary/mlx.core.array.flatten.html#mlx.core.array.flatten "mlx.core.array.flatten")(self[, start\_axis, end\_axis, ...]) | See [`flatten()`](_autosummary/mlx.core.flatten.html#mlx.core.flatten "mlx.core.flatten"). |
| [`array.log`](_autosummary/mlx.core.array.log.html#mlx.core.array.log "mlx.core.array.log")(self, \*[, stream]) | See [`log()`](_autosummary/mlx.core.log.html#mlx.core.log "mlx.core.log"). |
| [`array.log10`](_autosummary/mlx.core.array.log10.html#mlx.core.array.log10 "mlx.core.array.log10")(self, \*[, stream]) | See [`log10()`](_autosummary/mlx.core.log10.html#mlx.core.log10 "mlx.core.log10"). |
| [`array.log1p`](_autosummary/mlx.core.array.log1p.html#mlx.core.array.log1p "mlx.core.array.log1p")(self, \*[, stream]) | See [`log1p()`](_autosummary/mlx.core.log1p.html#mlx.core.log1p "mlx.core.log1p"). |
| [`array.log2`](_autosummary/mlx.core.array.log2.html#mlx.core.array.log2 "mlx.core.array.log2")(self, \*[, stream]) | See [`log2()`](_autosummary/mlx.core.log2.html#mlx.core.log2 "mlx.core.log2"). |
| [`array.logcumsumexp`](_autosummary/mlx.core.array.logcumsumexp.html#mlx.core.array.logcumsumexp "mlx.core.array.logcumsumexp")(self[, axis, reverse, ...]) | See [`logcumsumexp()`](_autosummary/mlx.core.logcumsumexp.html#mlx.core.logcumsumexp "mlx.core.logcumsumexp"). |
| [`array.logsumexp`](_autosummary/mlx.core.array.logsumexp.html#mlx.core.array.logsumexp "mlx.core.array.logsumexp")(self[, axis, keepdims, stream]) | See [`logsumexp()`](_autosummary/mlx.core.logsumexp.html#mlx.core.logsumexp "mlx.core.logsumexp"). |
| [`array.max`](_autosummary/mlx.core.array.max.html#mlx.core.array.max "mlx.core.array.max")(self[, axis, keepdims, stream]) | See [`max()`](_autosummary/mlx.core.max.html#mlx.core.max "mlx.core.max"). |
| [`array.mean`](_autosummary/mlx.core.array.mean.html#mlx.core.array.mean "mlx.core.array.mean")(self[, axis, keepdims, stream]) | See [`mean()`](_autosummary/mlx.core.mean.html#mlx.core.mean "mlx.core.mean"). |
| [`array.min`](_autosummary/mlx.core.array.min.html#mlx.core.array.min "mlx.core.array.min")(self[, axis, keepdims, stream]) | See [`min()`](_autosummary/mlx.core.min.html#mlx.core.min "mlx.core.min"). |
| [`array.moveaxis`](_autosummary/mlx.core.array.moveaxis.html#mlx.core.array.moveaxis "mlx.core.array.moveaxis")(self, source, destination, \*) | See [`moveaxis()`](_autosummary/mlx.core.moveaxis.html#mlx.core.moveaxis "mlx.core.moveaxis"). |
| [`array.prod`](_autosummary/mlx.core.array.prod.html#mlx.core.array.prod "mlx.core.array.prod")(self[, axis, keepdims, stream]) | See [`prod()`](_autosummary/mlx.core.prod.html#mlx.core.prod "mlx.core.prod"). |
| [`array.reciprocal`](_autosummary/mlx.core.array.reciprocal.html#mlx.core.array.reciprocal "mlx.core.array.reciprocal")(self, \*[, stream]) | See [`reciprocal()`](_autosummary/mlx.core.reciprocal.html#mlx.core.reciprocal "mlx.core.reciprocal"). |
| [`array.reshape`](_autosummary/mlx.core.array.reshape.html#mlx.core.array.reshape "mlx.core.array.reshape")(self, \*shape[, stream]) | Equivalent to [`reshape()`](_autosummary/mlx.core.reshape.html#mlx.core.reshape "mlx.core.reshape") but the shape can be passed either as a [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)") or as separate arguments. |
| [`array.round`](_autosummary/mlx.core.array.round.html#mlx.core.array.round "mlx.core.array.round")(self[, decimals, stream]) | See [`round()`](_autosummary/mlx.core.round.html#mlx.core.round "mlx.core.round"). |
| [`array.rsqrt`](_autosummary/mlx.core.array.rsqrt.html#mlx.core.array.rsqrt "mlx.core.array.rsqrt")(self, \*[, stream]) | See [`rsqrt()`](_autosummary/mlx.core.rsqrt.html#mlx.core.rsqrt "mlx.core.rsqrt"). |
| [`array.sin`](_autosummary/mlx.core.array.sin.html#mlx.core.array.sin "mlx.core.array.sin")(self, \*[, stream]) | See [`sin()`](_autosummary/mlx.core.sin.html#mlx.core.sin "mlx.core.sin"). |
| [`array.split`](_autosummary/mlx.core.array.split.html#mlx.core.array.split "mlx.core.array.split")(self, indices\_or\_sections[, ...]) | See [`split()`](_autosummary/mlx.core.split.html#mlx.core.split "mlx.core.split"). |
| [`array.sqrt`](_autosummary/mlx.core.array.sqrt.html#mlx.core.array.sqrt "mlx.core.array.sqrt")(self, \*[, stream]) | See [`sqrt()`](_autosummary/mlx.core.sqrt.html#mlx.core.sqrt "mlx.core.sqrt"). |
| [`array.square`](_autosummary/mlx.core.array.square.html#mlx.core.array.square "mlx.core.array.square")(self, \*[, stream]) | See [`square()`](_autosummary/mlx.core.square.html#mlx.core.square "mlx.core.square"). |
| [`array.squeeze`](_autosummary/mlx.core.array.squeeze.html#mlx.core.array.squeeze "mlx.core.array.squeeze")(self[, axis, stream]) | See [`squeeze()`](_autosummary/mlx.core.squeeze.html#mlx.core.squeeze "mlx.core.squeeze"). |
| [`array.std`](_autosummary/mlx.core.array.std.html#mlx.core.array.std "mlx.core.array.std")(self[, axis, keepdims, ddof, stream]) | See [`std()`](_autosummary/mlx.core.std.html#mlx.core.std "mlx.core.std"). |
| [`array.sum`](_autosummary/mlx.core.array.sum.html#mlx.core.array.sum "mlx.core.array.sum")(self[, axis, keepdims, stream]) | See [`sum()`](_autosummary/mlx.core.sum.html#mlx.core.sum "mlx.core.sum"). |
| [`array.swapaxes`](_autosummary/mlx.core.array.swapaxes.html#mlx.core.array.swapaxes "mlx.core.array.swapaxes")(self, axis1, axis2, \*[, stream]) | See [`swapaxes()`](_autosummary/mlx.core.swapaxes.html#mlx.core.swapaxes "mlx.core.swapaxes"). |
| [`array.transpose`](_autosummary/mlx.core.array.transpose.html#mlx.core.array.transpose "mlx.core.array.transpose")(self, \*axes[, stream]) | Equivalent to [`transpose()`](_autosummary/mlx.core.transpose.html#mlx.core.transpose "mlx.core.transpose") but the axes can be passed either as a tuple or as separate arguments. |
| [`array.T`](_autosummary/mlx.core.array.T.html#mlx.core.array.T "mlx.core.array.T") | Equivalent to calling `self.transpose()` with no arguments. |
| [`array.var`](_autosummary/mlx.core.array.var.html#mlx.core.array.var "mlx.core.array.var")(self[, axis, keepdims, ddof, stream]) | See [`var()`](_autosummary/mlx.core.var.html#mlx.core.var "mlx.core.var"). |
| [`array.view`](_autosummary/mlx.core.array.view.html#mlx.core.array.view "mlx.core.array.view")(self, dtype, \*[, stream]) | See [`view()`](_autosummary/mlx.core.view.html#mlx.core.view "mlx.core.view"). |