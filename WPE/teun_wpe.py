import functools
import operator

import click
import numpy as np


def get_working_shape(shape):
    "Flattens all but the last two dimension."
    product = functools.reduce(operator.mul, [1] + list(shape[:-2]))
    return [product] + list(shape[-2:])


def segment_axis(
        x,
        length,
        shift,
        axis=-1,
        end='cut',  # in ['pad', 'cut', None]
        pad_mode='constant',
        pad_value=0,
):

    """Generate a new array that chops the given array along the given axis
     into overlapping frames.

    Note: if end='pad' the return is maybe a copy

    Args:
        x: The array to segment
        length: The length of each frame
        shift: The number of array elements by which to step forward
               Negative values are also allowed.
        axis: The axis to operate on; if None, act on the flattened array
        end: What to do with the last frame, if the array is not evenly
                divisible into pieces. Options are:
                * 'cut'   Simply discard the extra values
                * None    No end treatment. Only works when fits perfectly.
                * 'pad'   Pad with a constant value
                * 'conv_pad' Special padding for convolution, assumes
                             shift == 1, see example below
        pad_mode: see numpy.pad
        pad_value: The value to use for end='pad'

    Examples:
        >>> # import cupy as np
        >>> segment_axis(np.arange(10), 4, 2)  # simple example
        array([[0, 1, 2, 3],
               [2, 3, 4, 5],
               [4, 5, 6, 7],
               [6, 7, 8, 9]])
        >>> segment_axis(np.arange(10), 4, -2)  # negative shift
        array([[6, 7, 8, 9],
               [4, 5, 6, 7],
               [2, 3, 4, 5],
               [0, 1, 2, 3]])
        >>> segment_axis(np.arange(5).reshape(5), 4, 1, axis=0)
        array([[0, 1, 2, 3],
               [1, 2, 3, 4]])
        >>> segment_axis(np.arange(5).reshape(5), 4, 2, axis=0, end='cut')
        array([[0, 1, 2, 3]])
        >>> segment_axis(np.arange(5).reshape(5), 4, 2, axis=0, end='pad')
        array([[0, 1, 2, 3],
               [2, 3, 4, 0]])
        >>> segment_axis(np.arange(5).reshape(5), 4, 1, axis=0, end='conv_pad')
        array([[0, 0, 0, 0],
               [0, 0, 0, 1],
               [0, 0, 1, 2],
               [0, 1, 2, 3],
               [1, 2, 3, 4],
               [2, 3, 4, 0],
               [3, 4, 0, 0],
               [4, 0, 0, 0]])
        >>> segment_axis(np.arange(6).reshape(6), 4, 2, axis=0, end='pad')
        array([[0, 1, 2, 3],
               [2, 3, 4, 5]])
        >>> segment_axis(np.arange(10).reshape(2, 5), 4, 1, axis=-1)
        array([[[0, 1, 2, 3],
                [1, 2, 3, 4]],
        <BLANKLINE>
               [[5, 6, 7, 8],
                [6, 7, 8, 9]]])
        >>> segment_axis(np.arange(10).reshape(5, 2).T, 4, 1, axis=1)
        array([[[0, 2, 4, 6],
                [2, 4, 6, 8]],
        <BLANKLINE>
               [[1, 3, 5, 7],
                [3, 5, 7, 9]]])
        >>> segment_axis(np.asfortranarray(np.arange(10).reshape(2, 5)),
        ...                 4, 1, axis=1)
        array([[[0, 1, 2, 3],
                [1, 2, 3, 4]],
        <BLANKLINE>
               [[5, 6, 7, 8],
                [6, 7, 8, 9]]])
        >>> segment_axis(np.arange(8).reshape(2, 2, 2).transpose(1, 2, 0),
        ...                 2, 1, axis=0, end='cut')
        array([[[[0, 4],
                 [1, 5]],
        <BLANKLINE>
                [[2, 6],
                 [3, 7]]]])
        >>> a = np.arange(7).reshape(7)
        >>> b = segment_axis(a, 4, -2, axis=0, end='cut')
        >>> a += 1  # a and b point to the same memory
        >>> b
        array([[3, 4, 5, 6],
               [1, 2, 3, 4]])

        >>> segment_axis(np.arange(7), 8, 1, axis=0, end='pad').shape
        (1, 8)
        >>> segment_axis(np.arange(8), 8, 1, axis=0, end='pad').shape
        (1, 8)
        >>> segment_axis(np.arange(9), 8, 1, axis=0, end='pad').shape
        (2, 8)
        >>> segment_axis(np.arange(7), 8, 2, axis=0, end='cut').shape
        (0, 8)
        >>> segment_axis(np.arange(8), 8, 2, axis=0, end='cut').shape
        (1, 8)
        >>> segment_axis(np.arange(9), 8, 2, axis=0, end='cut').shape
        (1, 8)

        >>> x = np.arange(1, 10)
        >>> filter_ = np.array([1, 2, 3])
        >>> np.convolve(x, filter_)
        array([ 1,  4, 10, 16, 22, 28, 34, 40, 46, 42, 27])
        >>> x_ = segment_axis(x, len(filter_), 1, end='conv_pad')
        >>> x_
        array([[0, 0, 1],
               [0, 1, 2],
               [1, 2, 3],
               [2, 3, 4],
               [3, 4, 5],
               [4, 5, 6],
               [5, 6, 7],
               [6, 7, 8],
               [7, 8, 9],
               [8, 9, 0],
               [9, 0, 0]])
        >>> x_ @ filter_[::-1]  # Equal to convolution
        array([ 1,  4, 10, 16, 22, 28, 34, 40, 46, 42, 27])

        >>> segment_axis(np.arange(19), 16, 4, axis=-1, end='pad')
        array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
               [ 4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,  0]])

        >>> import torch
        >>> segment_axis(torch.tensor(np.arange(10)), 4, 2)  # simple example
        tensor([[0, 1, 2, 3],
                [2, 3, 4, 5],
                [4, 5, 6, 7],
                [6, 7, 8, 9]])
    """
    backend = {
        'numpy': 'numpy',
        'cupy.core.core': 'cupy',
        'torch': 'torch',
    }[x.__class__.__module__]

    if backend == 'numpy':
        xp = np
    elif backend == 'cupy':
        import cupy
        xp = cupy
    elif backend == 'torch':
        import torch
        xp = torch
    else:
        raise Exception('Can not happen')

    try:
        ndim = x.ndim
    except AttributeError:
        # For Pytorch 1.2 and below
        ndim = x.dim()

    axis = axis % ndim

    # Implement negative shift with a positive shift and a flip
    # stride_tricks does not work correct with negative stride
    if shift > 0:
        do_flip = False
    elif shift < 0:
        do_flip = True
        shift = abs(shift)
    else:
        raise ValueError(shift)

    if pad_mode == 'constant':
        pad_kwargs = {'constant_values': pad_value}
    else:
        pad_kwargs = {}

    # Pad
    if end == 'pad':
        if x.shape[axis] < length:
            npad = np.zeros([ndim, 2], dtype=np.int)
            npad[axis, 1] = length - x.shape[axis]
            x = xp.pad(x, pad_width=npad, mode=pad_mode, **pad_kwargs)
        elif shift != 1 and (x.shape[axis] + shift - length) % shift != 0:
            npad = np.zeros([ndim, 2], dtype=np.int)
            npad[axis, 1] = shift - ((x.shape[axis] + shift - length) % shift)
            x = xp.pad(x, pad_width=npad, mode=pad_mode, **pad_kwargs)

    elif end == 'conv_pad':
        assert shift == 1, shift
        npad = np.zeros([ndim, 2], dtype=np.int)
        npad[axis, :] = length - shift
        x = xp.pad(x, pad_width=npad, mode=pad_mode, **pad_kwargs)
    elif end is None:
        assert (x.shape[axis] + shift - length) % shift == 0, \
            '{} = x.shape[axis]({}) + shift({}) - length({})) % shift({})' \
            ''.format((x.shape[axis] + shift - length) % shift,
                      x.shape[axis], shift, length, shift)
    elif end == 'cut':
        pass
    else:
        raise ValueError(end)

    # Calculate desired shape and strides
    shape = list(x.shape)
    # assert shape[axis] >= length, shape
    del shape[axis]
    shape.insert(axis, (x.shape[axis] + shift - length) // shift)
    shape.insert(axis + 1, length)

    def get_strides(array):
        try:
            return list(array.strides)
        except AttributeError:
            # fallback for torch
            return list(array.stride())

    strides = get_strides(x)
    strides.insert(axis, shift * strides[axis])

    # Alternative to np.ndarray.__new__
    # I am not sure if np.lib.stride_tricks.as_strided is better.
    # return np.lib.stride_tricks.as_strided(
    #     x, shape=shape, strides=strides)
    try:
        if backend == 'numpy':
            x = np.lib.stride_tricks.as_strided(x, strides=strides, shape=shape)
        elif backend == 'cupy':
            x = x.view()
            x._set_shape_and_strides(strides=strides, shape=shape)
        elif backend == 'torch':
            import torch
            x = torch.as_strided(x, size=shape, stride=strides)
        else:
            raise Exception('Can not happen')

        # return np.ndarray.__new__(np.ndarray, strides=strides,
        #                           shape=shape, buffer=x, dtype=x.dtype)
    except Exception:
        print('strides:', get_strides(x), ' -> ', strides)
        print('shape:', x.shape, ' -> ', shape)
        try:
            print('flags:', x.flags)
        except AttributeError:
            pass  # for pytorch
        print('Parameters:')
        print('shift:', shift, 'Note: negative shift is implemented with a '
                               'following flip')
        print('length:', length, '<- Has to be positive.')
        raise
    if do_flip:
        return xp.flip(x, axis=axis)
    else:
        return x

def _lstsq(A, B):
    assert A.shape == B.shape, (A.shape, B.shape)
    shape = A.shape

    working_shape = get_working_shape(shape)

    A = A.reshape(working_shape)
    B = B.reshape(working_shape)

    C = np.zeros_like(A)
    for i in range(working_shape[0]):
        C[i] = np.linalg.lstsq(A[i], B[i])[0]
    return C.reshape(*shape)


def _stable_solve(A, B):
    """
    Use np.linalg.solve with fallback to np.linalg.lstsq.
    Equal to np.linalg.lstsq but faster.

    Note: limited currently by A.shape == B.shape

    This function try's np.linalg.solve with independent dimensions,
    when this is not working the function fall back to np.linalg.solve
    for each matrix. If one matrix does not work it fall back to
    np.linalg.lstsq.

    The reason for not using np.linalg.lstsq directly is the execution time.
    Examples:
    A and B have the shape (500, 6, 6), than a loop over lstsq takes
    108 ms and this function 28 ms for the case that one matrix is singular
    else 1 ms.

    >>> def normal(shape):
    ...     return np.random.normal(size=shape) + 1j * np.random.normal(size=shape)

    >>> A = normal((6, 6))
    >>> B = normal((6, 6))
    >>> C1 = np.linalg.solve(A, B)
    >>> C2, *_ = np.linalg.lstsq(A, B)
    >>> C3 = _stable_solve(A, B)
    >>> C4 = _lstsq(A, B)
    >>> np.testing.assert_allclose(C1, C2)
    >>> np.testing.assert_allclose(C1, C3)
    >>> np.testing.assert_allclose(C1, C4)

    >>> A = np.zeros((6, 6), dtype=np.complex128)
    >>> B = np.zeros((6, 6), dtype=np.complex128)
    >>> C1 = np.linalg.solve(A, B)
    Traceback (most recent call last):
    ...
    numpy.linalg.linalg.LinAlgError: Singular matrix
    >>> C2, *_ = np.linalg.lstsq(A, B)
    >>> C3 = _stable_solve(A, B)
    >>> C4 = _lstsq(A, B)
    >>> np.testing.assert_allclose(C2, C3)
    >>> np.testing.assert_allclose(C2, C4)

    >>> A = normal((3, 6, 6))
    >>> B = normal((3, 6, 6))
    >>> C1 = np.linalg.solve(A, B)
    >>> C2, *_ = np.linalg.lstsq(A, B)
    Traceback (most recent call last):
    ...
    numpy.linalg.linalg.LinAlgError: 3-dimensional array given. Array must be two-dimensional
    >>> C3 = _stable_solve(A, B)
    >>> C4 = _lstsq(A, B)
    >>> np.testing.assert_allclose(C1, C3)
    >>> np.testing.assert_allclose(C1, C4)


    >>> A[2, 3, :] = 0
    >>> C1 = np.linalg.solve(A, B)
    Traceback (most recent call last):
    ...
    numpy.linalg.linalg.LinAlgError: Singular matrix
    >>> C2, *_ = np.linalg.lstsq(A, B)
    Traceback (most recent call last):
    ...
    numpy.linalg.linalg.LinAlgError: 3-dimensional array given. Array must be two-dimensional
    >>> C3 = _stable_solve(A, B)
    >>> C4 = _lstsq(A, B)
    >>> np.testing.assert_allclose(C3, C4)


    """
    assert A.shape[:-2] == B.shape[:-2], (A.shape, B.shape)
    assert A.shape[-1] == B.shape[-2], (A.shape, B.shape)
    try:
        return np.linalg.solve(A, B)
    except np.linalg.linalg.LinAlgError:
        shape_A, shape_B = A.shape, B.shape
        assert shape_A[:-2] == shape_A[:-2]
        working_shape_A = get_working_shape(shape_A)
        working_shape_B = get_working_shape(shape_B)
        A = A.reshape(working_shape_A)
        B = B.reshape(working_shape_B)

        C = np.zeros_like(B)
        for i in range(working_shape_A[0]):
            # lstsq is much slower, use it only when necessary
            try:
                C[i] = np.linalg.solve(A[i], B[i])
            except np.linalg.linalg.LinAlgError:
                C[i] = np.linalg.lstsq(A[i], B[i])[0]
        return C.reshape(*shape_B)


def build_y_tilde(Y, taps, delay):
    """

    Note: The returned y_tilde consumes a similar amount of memory as Y, because
        of tricks with strides. Usually the memory consumprion is K times
        smaller than the memory consumprion of a contignous array,

    >>> T, D = 20, 2
    >>> Y = np.arange(start=1, stop=T * D + 1).reshape([T, D]).T
    >>> print(Y)
    [[ 1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39]
     [ 2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40]]
    >>> taps, delay = 4, 2
    >>> Y_tilde = build_y_tilde(Y, taps, delay)
    >>> print(Y_tilde.shape, (taps*D, T))
    (8, 20) (8, 20)
    >>> print(Y_tilde)
    [[ 0  0  1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31 33 35]
     [ 0  0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34 36]
     [ 0  0  0  1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31 33]
     [ 0  0  0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34]
     [ 0  0  0  0  1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31]
     [ 0  0  0  0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32]
     [ 0  0  0  0  0  1  3  5  7  9 11 13 15 17 19 21 23 25 27 29]
     [ 0  0  0  0  0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30]]
    >>> Y_tilde = build_y_tilde(Y, taps, 0)
    >>> print(Y_tilde.shape, (taps*D, T), Y_tilde.strides)
    (8, 20) (8, 20) (-8, 16)
    >>> print('Pseudo size:', Y_tilde.nbytes)
    Pseudo size: 1280
    >>> print('Reak size:', Y_tilde.base.base.base.base.nbytes)
    Reak size: 368
    >>> print(Y_tilde)
    [[ 1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39]
     [ 2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40]
     [ 0  1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31 33 35 37]
     [ 0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38]
     [ 0  0  1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31 33 35]
     [ 0  0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34 36]
     [ 0  0  0  1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31 33]
     [ 0  0  0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34]]

    The first columns are zero because of the delay.

    """
    S = Y.shape[:-2]
    D = Y.shape[-2]
    T = Y.shape[-1]

    def pad(x, axis=-1, pad_width=taps + delay - 1):
        npad = np.zeros([x.ndim, 2], dtype=np.int)
        npad[axis, 0] = pad_width
        x = np.pad(x,
                   pad_width=npad,
                   mode='constant',
                   constant_values=0)
        return x

    # Y_ = segment_axis(pad(Y), K, 1, axis=-1)
    # Y_ = np.flip(Y_, axis=-1)
    # if delay > 0:
    #     Y_ = Y_[..., :-delay, :]
    # # Y_: ... x D x T x K
    # Y_ = np.moveaxis(Y_, -1, -3)
    # # Y_: ... x K x D x T
    # Y_ = np.reshape(Y_, [*S, K * D, T])
    # # Y_: ... x KD x T

    # ToDo: write the shape
    Y_ = pad(Y)
    Y_ = np.moveaxis(Y_, -1, -2)
    Y_ = np.flip(Y_, axis=-1)
    Y_ = np.ascontiguousarray(Y_)
    Y_ = np.flip(Y_, axis=-1)
    Y_ = segment_axis(Y_, taps, 1, axis=-2)
    Y_ = np.flip(Y_, axis=-2)
    if delay > 0:
        Y_ = Y_[..., :-delay, :, :]
    Y_ = np.reshape(Y_, list(S) + [T, taps * D])
    Y_ = np.moveaxis(Y_, -2, -1)

    return Y_


def hermite(x):
    return x.swapaxes(-2, -1).conj()

def abs_square(x):
    """

    Params:
        x: np.ndarray

    https://github.com/numpy/numpy/issues/9679

    Bug in numpy 1.13.1
    >> np.ones(32768).imag ** 2
    Traceback (most recent call last):
    ...
    ValueError: output array is read-only
    >> np.ones(32767).imag ** 2
    array([ 0.,  0.,  0., ...,  0.,  0.,  0.])

    >>> abs_square(np.ones(32768)).shape
    (32768,)
    >>> abs_square(np.ones(32768, dtype=np.complex64)).shape
    (32768,)
    """

    if np.iscomplexobj(x):
        return x.real ** 2 + x.imag ** 2
    else:
        return x ** 2


def window_mean(x, lr_context, axis=-1):
    """
    Take the mean of x at each index with a left and right context.
    Pseudo code for lr_context == (1, 1):
        y = np.zeros(...)
        for i in range(...):
            if not edge_case(i):
                y[i] = (x[i - 1] + x[i] + x[i + 1]) / 3
            elif i == 0:
                y[i] = (x[i] + x[i + 1]) / 2
            else:
                y[i] = (x[i - 1] + x[i]) / 2
        return y

    >>> window_mean([1, 1, 1, 1, 1], 1)
    array([1., 1., 1., 1., 1.])
    >>> window_mean([1, 2, 3, 4, 5], 1)
    array([1.5, 2. , 3. , 4. , 4.5])
    >>> x = [1, 1, 13, 1, 1]
    >>> np.testing.assert_equal(window_mean(x, (0, 1)), [1, 7, 7, 1, 1])
    >>> np.testing.assert_equal(window_mean(x, (1, 0)), [1, 1, 7, 7, 1])
    >>> np.testing.assert_equal(window_mean(x, (0, 2)), [5, 5, 5, 1, 1])
    >>> np.testing.assert_equal(window_mean(x, (2, 0)), [1, 1, 5, 5, 5])
    >>> np.testing.assert_equal(window_mean(x, (1, 2)), [5, 4, 4, 5, 1])
    >>> np.testing.assert_equal(window_mean(x, (2, 1)), [1, 5, 4, 4, 5])
    >>> np.testing.assert_equal(window_mean(x, (9, 9)), [3.4] * 5)

    >>> x = np.random.normal(size=(20, 50))
    >>> lr_context = np.random.randint(0, 5, size=2)
    >>> a = window_mean(x, lr_context, axis=1)
    >>> b = window_mean(x, lr_context, axis=-1)
    >>> c = window_mean(x.T, lr_context, axis=0).T
    >>> d = [window_mean_slow(s, lr_context) for s in x]
    >>> np.testing.assert_equal(a, b)
    >>> np.testing.assert_equal(a, c)
    >>> np.testing.assert_almost_equal(a, d)

    >>> import bottleneck as bn
    >>> a = window_mean(x, [lr_context[0], 0], axis=-1)
    >>> b = bn.move_mean(x, lr_context[0] + 1, min_count=1)
    >>> np.testing.assert_almost_equal(a, b)

    >>> a = window_mean(x, [lr_context[0], 0], axis=0)
    >>> b = bn.move_mean(x, lr_context[0] + 1, min_count=1, axis=0)
    >>> np.testing.assert_almost_equal(a, b)

    """
    if isinstance(lr_context, int):
        lr_context = [lr_context + 1, lr_context]
    else:
        assert len(lr_context) == 2, lr_context
        tmp_l_context, tmp_r_context = lr_context
        lr_context = tmp_l_context + 1, tmp_r_context

    x = np.asarray(x)

    window_length = sum(lr_context)
    if window_length == 0:
        return x

    pad_width = np.zeros((x.ndim, 2), dtype=np.int64)
    pad_width[axis] = lr_context

    first_slice = [slice(None)] * x.ndim
    first_slice[axis] = slice(sum(lr_context), None)
    second_slice = [slice(None)] * x.ndim
    second_slice[axis] = slice(None, -sum(lr_context))

    def foo(x):
        cumsum = np.cumsum(np.pad(x, pad_width, mode='constant'), axis=axis)
        return cumsum[first_slice] - cumsum[second_slice]

    ones_shape = [1] * x.ndim
    ones_shape[axis] = x.shape[axis]

    return foo(x) / foo(np.ones(ones_shape, np.int64))



def _stable_positive_inverse(power):
    """
    Calculate the inverse of a positive value.
    """
    eps = 1e-10 * np.max(power)
    if eps == 0:
        # Special case when signal is zero.
        # Does not happen on real data.
        # This only happens in artificial cases, e.g. redacted signal parts,
        # where the signal is set to be zero from a human.
        #
        # The scale of the power does not matter, so take 1.
        inverse_power = np.ones_like(power)
    else:
        inverse_power = 1 / np.maximum(power, eps)
    return inverse_power


def get_power_inverse(signal, psd_context=0):
    """
    Assumes single frequency bin with shape (D, T).

    >>> s = 1 / np.array([np.arange(1, 6)]*3)
    >>> get_power_inverse(s)
    array([ 1.,  4.,  9., 16., 25.])
    >>> get_power_inverse(s * 0 + 1, 1)
    array([1., 1., 1., 1., 1.])
    >>> get_power_inverse(s, 1)
    array([ 1.6       ,  2.20408163,  7.08196721, 14.04421326, 19.51219512])
    >>> get_power_inverse(s, np.inf)
    array([3.41620801, 3.41620801, 3.41620801, 3.41620801, 3.41620801])
    >>> get_power_inverse(s * 0.)
    array([1., 1., 1., 1., 1.])
    """
    power = abs_square(signal)

    if np.isposinf(psd_context):
        power = np.broadcast_to(np.mean(power, axis=-1, keepdims=True), power.shape)
    elif psd_context > 0:
        assert int(psd_context) == psd_context, psd_context
        psd_context = int(psd_context)
        # import bottleneck as bn
        # Handle the corner case correctly (i.e. sum() / count)
        # Use bottleneck when only left context is requested
        # power = bn.move_mean(power, psd_context*2+1, min_count=1)
        power = window_mean(power, (psd_context, psd_context))
    elif psd_context == 0:
        pass
    else:
        raise ValueError(psd_context)
    return _stable_positive_inverse(power)


def get_correlations_v12(Y, Y_tilde, power):

    inverse_power = _stable_positive_inverse(3*power)[:, 0, :].reshape(power.shape[0],1,power.shape[2])
    power = power[:, 0, :].reshape(power.shape[0],1,power.shape[2])
    Y_tilde_inverse_power = (2 * np.pi * (abs_square(Y_tilde) + power)**1.5)*inverse_power
    R = np.matmul(Y_tilde_inverse_power, hermite(Y_tilde_inverse_power))
    P = np.matmul(Y_tilde_inverse_power, hermite(Y))
    return R, P

def get_correlations_v13(Y, Y_tilde, power):
    inverse_power = _stable_positive_inverse(3*power)[:, 0, :].reshape(power.shape[0],1,power.shape[2])
    power = power[:, 0, :].reshape(power.shape[0],1,power.shape[2])
    Y_tilde_inverse_power = ((abs_square(Y_tilde) + power))*inverse_power

    R = np.matmul(Y_tilde_inverse_power, hermite(Y_tilde_inverse_power))
    P = np.matmul(Y_tilde_inverse_power, hermite(Y))
    return R, P

def get_correlations_v14(Y, Y_tilde, power):
    power = power[:, 0, :].reshape(power.shape[0],1,power.shape[2])
    Y_tilde_inverse_power = power /  (3 * power) * _stable_positive_inverse((abs_square(Y_tilde) + power))

    R = np.matmul(Y_tilde_inverse_power, hermite(Y_tilde_inverse_power))
    P = np.matmul(Y_tilde_inverse_power, hermite(Y))
    return R, P


def get_filter_matrix_v12(Y, Y_tilde, inverse_power):
    R, P = get_correlations_v12(Y, Y_tilde, inverse_power)
    G = _stable_solve(R, P)
    return G

def get_filter_matrix_v13(Y, Y_tilde, inverse_power):
    R, P = get_correlations_v13(Y, Y_tilde, inverse_power)
    G = _stable_solve(R, P)
    return G

def get_filter_matrix_v14(Y, Y_tilde, inverse_power):
    R, P = get_correlations_v14(Y, Y_tilde, inverse_power)
    G = _stable_solve(R, P)
    return G


def perform_filter_operation_v5(Y, Y_tilde, filter_matrix):
    X = Y - np.matmul(hermite(filter_matrix), Y_tilde)
    return X

def wpe_v12(Y, taps=10, delay=3, iterations=3, psd_context=0, statistics_mode='full'):
    """
    Batched and modular WPE version.

    Args:
        Y: Complex valued STFT signal with shape (..., D, T).
        taps: Filter order
        delay: Delay as a guard interval, such that X does not become zero.
        iterations:
        psd_context: Defines the number of elements in the time window
            to improve the power estimation. Total number of elements will
            be (psd_context + 1 + psd_context).
        statistics_mode: Either 'full' or 'valid'.
            'full': Pad the observation with zeros on the left for the
            estimation of the correlation matrix and vector.
            'valid': Only calculate correlation matrix and vector on valid
            slices of the observation.

    Returns:
        Estimated signal with the same shape as Y
    """
    X = Y
    Y_tilde = build_y_tilde(Y, taps, delay)

    if statistics_mode == 'full':
        s = Ellipsis
    elif statistics_mode == 'valid':
        s = (Ellipsis, slice(delay + taps - 1, None))
    else:
        raise ValueError(statistics_mode)

    for iteration in range(iterations):
        # inverse_power = get_power_inverse(X, psd_context=psd_context)
        power = abs_square(X)
        G = get_filter_matrix_v12(Y=Y[s], Y_tilde=Y_tilde[s], inverse_power=power[s])
        X = perform_filter_operation_v5(Y=Y, Y_tilde=Y_tilde, filter_matrix=G)
    return X


def wpe_v13(Y, taps=10, delay=3, iterations=3, psd_context=0, statistics_mode='full'):
    """
    Batched and modular WPE version.

    Args:
        Y: Complex valued STFT signal with shape (..., D, T).
        taps: Filter order
        delay: Delay as a guard interval, such that X does not become zero.
        iterations:
        psd_context: Defines the number of elements in the time window
            to improve the power estimation. Total number of elements will
            be (psd_context + 1 + psd_context).
        statistics_mode: Either 'full' or 'valid'.
            'full': Pad the observation with zeros on the left for the
            estimation of the correlation matrix and vector.
            'valid': Only calculate correlation matrix and vector on valid
            slices of the observation.

    Returns:
        Estimated signal with the same shape as Y
    """
    X = Y
    Y_tilde = build_y_tilde(Y, taps, delay)

    if statistics_mode == 'full':
        s = Ellipsis
    elif statistics_mode == 'valid':
        s = (Ellipsis, slice(delay + taps - 1, None))
    else:
        raise ValueError(statistics_mode)

    for iteration in range(iterations):
        # inverse_power = get_power_inverse(X, psd_context=psd_context)
        power = abs_square(X)
        G = get_filter_matrix_v13(Y=Y[s], Y_tilde=Y_tilde[s], inverse_power=power[s])
        X = perform_filter_operation_v5(Y=Y, Y_tilde=Y_tilde, filter_matrix=G)
    return X

def wpe_v14(Y, taps=10, delay=3, iterations=3, psd_context=0, statistics_mode='full'):
    """
    Batched and modular WPE version.

    Args:
        Y: Complex valued STFT signal with shape (..., D, T).
        taps: Filter order
        delay: Delay as a guard interval, such that X does not become zero.
        iterations:
        psd_context: Defines the number of elements in the time window
            to improve the power estimation. Total number of elements will
            be (psd_context + 1 + psd_context).
        statistics_mode: Either 'full' or 'valid'.
            'full': Pad the observation with zeros on the left for the
            estimation of the correlation matrix and vector.
            'valid': Only calculate correlation matrix and vector on valid
            slices of the observation.

    Returns:
        Estimated signal with the same shape as Y
    """
    X = Y
    Y_tilde = build_y_tilde(Y, taps, delay)

    if statistics_mode == 'full':
        s = Ellipsis
    elif statistics_mode == 'valid':
        s = (Ellipsis, slice(delay + taps - 1, None))
    else:
        raise ValueError(statistics_mode)

    for iteration in range(iterations):
        # inverse_power = get_power_inverse(X, psd_context=psd_context)
        power = abs_square(X)
        G = get_filter_matrix_v14(Y=Y[s], Y_tilde=Y_tilde[s], inverse_power=power[s])
        X = perform_filter_operation_v5(Y=Y, Y_tilde=Y_tilde, filter_matrix=G)
    return X
