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