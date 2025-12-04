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