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