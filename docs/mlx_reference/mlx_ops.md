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