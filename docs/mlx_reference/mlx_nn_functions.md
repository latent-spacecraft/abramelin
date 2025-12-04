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