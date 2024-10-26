# Copied from: https://raw.githubusercontent.com/jonathf/chaospy/master/chaospy/distributions/sampler/sequences/


"""
Create samples from the `Hammersley set`_.
The Hammersley set is equivalent to the Halton sequence, except for one
dimension is replaced with a regular grid.
Example usage
-------------
Standard usage::
    >>> distribution = chaospy.J(chaospy.Uniform(0, 1), chaospy.Uniform(0, 1))
    >>> samples = distribution.sample(3, rule="M")
    >>> print(numpy.around(samples, 4))
    [[0.75  0.125 0.625]
     [0.25  0.5   0.75 ]]
    >>> samples = distribution.sample(4, rule="M")
    >>> print(numpy.around(samples, 4))
    [[0.75  0.125 0.625 0.375]
     [0.2   0.4   0.6   0.8  ]]
.. _Hammersley set: https://en.wikipedia.org/wiki/Low-discrepancy_sequence#Hammersley_set
"""

import numpy

# from .halton import create_halton_samples


def create_hammersley_samples(order, dim=1, burnin=-1, primes=(), **kwargs):
    """
    Create samples from the Hammersley set.
    For ``dim == 1`` the sequence falls back to Van Der Corput sequence.
    Args:
        order (int):
            The order of the Hammersley sequence. Defines the number of samples.
        dim (int):
            The number of dimensions in the Hammersley sequence.
        burnin (int):
            Skip the first ``burnin`` samples. If negative, the maximum of
            ``primes`` is used.
        primes (tuple):
            The (non-)prime base to calculate values along each axis. If
            empty, growing prime values starting from 2 will be used.
    Returns:
        (numpy.ndarray):
            Hammersley set with ``shape == (dim, order)``.
    """
    if dim == 1:
        return create_halton_samples(order=order, dim=1, burnin=burnin, primes=primes)
    out = numpy.empty((dim, order), dtype=float)
    out[: dim - 1] = create_halton_samples(
        order=order, dim=dim - 1, burnin=burnin, primes=primes
    )
    out[dim - 1] = numpy.linspace(0, 1, order + 2)[1:-1]
    return out


"""
Create samples from the `Halton sequence`_.

In statistics, Halton sequences are sequences used to generate points in space
for numerical methods such as Monte Carlo simulations. Although these sequences
are deterministic, they are of low discrepancy, that is, appear to be random
for many purposes. They were first introduced in 1960 and are an example of
a quasi-random number sequence. They generalise the one-dimensional van der
Corput sequences.

Example usage
-------------

Standard usage::

    >>> distribution = chaospy.J(chaospy.Uniform(0, 1), chaospy.Uniform(0, 1))
    >>> samples = distribution.sample(3, rule="H")
    >>> print(numpy.around(samples, 4))
    [[0.125  0.625  0.375 ]
     [0.4444 0.7778 0.2222]]
    >>> samples = distribution.sample(4, rule="H")
    >>> print(numpy.around(samples, 4))
    [[0.125  0.625  0.375  0.875 ]
     [0.4444 0.7778 0.2222 0.5556]]

.. _Halton sequence: https://en.wikipedia.org/wiki/Halton_sequence
"""


def create_halton_samples(order, dim=1, burnin=-1, primes=()):
    """
    Create Halton sequence.

    For ``dim == 1`` the sequence falls back to Van Der Corput sequence.

    Args:
        order (int):
            The order of the Halton sequence. Defines the number of samples.
        dim (int):
            The number of dimensions in the Halton sequence.
        burnin (int):
            Skip the first ``burnin`` samples. If negative, the maximum of
            ``primes`` is used.
        primes (tuple):
            The (non-)prime base to calculate values along each axis. If
            empty, growing prime values starting from 2 will be used.

    Returns (numpy.ndarray):
        Halton sequence with ``shape == (dim, order)``.
    """
    primes = list(primes)
    if not primes:
        prime_order = 10 * dim
        while len(primes) < dim:
            primes = create_primes(prime_order)
            prime_order *= 2
    primes = primes[:dim]
    assert len(primes) == dim, "not enough primes"

    if burnin < 0:
        burnin = max(primes)

    out = numpy.empty((dim, order))
    indices = [idx + burnin for idx in range(order)]
    for dim_ in range(dim):
        out[dim_] = create_van_der_corput_samples(indices, number_base=primes[dim_])
    return out


"""
Create `Van Der Corput` low discrepancy sequence samples.

A van der Corput sequence is an example of the simplest one-dimensional
low-discrepancy sequence over the unit interval; it was first described in 1935
by the Dutch mathematician J. G. van der Corput. It is constructed by reversing
the base-n representation of the sequence of natural numbers (1, 2, 3, ...).

In practice, use Halton sequence instead of Van Der Corput, as it is the
same, but generalized to work in multiple dimensions.

Example usage
-------------

Using base 10::

    >>> print(create_van_der_corput_samples(range(11), number_base=10))
    [0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  0.01 0.11]

Using base 2::

    >>> print(create_van_der_corput_samples(range(8), number_base=2))
    [0.5    0.25   0.75   0.125  0.625  0.375  0.875  0.0625]

.. Van Der Corput: https://en.wikipedia.org/wiki/Van_der_Corput_sequence
"""


def create_van_der_corput_samples(idx, number_base=2):
    """
    Van der Corput samples.

    Args:
        idx (int, numpy.ndarray):
            The index of the sequence. If array is provided, all values in
            array is returned.
        number_base (int):
            The numerical base from where to create the samples from.

    Returns (float, numpy.ndarray):
        Van der Corput samples.
    """
    assert number_base > 1

    idx = numpy.asarray(idx).flatten() + 1
    out = numpy.zeros(len(idx), dtype=float)

    base = float(number_base)
    active = numpy.ones(len(idx), dtype=bool)
    while numpy.any(active):
        out[active] += (idx[active] % number_base) / base
        idx //= number_base
        base *= number_base
        active = idx > 0
    return out


"""
Create all primes bellow a certain threshold.

Examples::

    >>> print(create_primes(1))
    []
    >>> print(create_primes(2))
    [2]
    >>> print(create_primes(3))
    [2, 3]
    >>> print(create_primes(20))
    [2, 3, 5, 7, 11, 13, 17, 19]
"""


def create_primes(threshold):
    """
    Generate prime values using sieve of Eratosthenes method.

    Args:
        threshold (int):
            The upper bound for the size of the prime values.

    Returns (List[int]):
        All primes from 2 and up to ``threshold``.
    """
    if threshold == 2:
        return [2]

    elif threshold < 2:
        return []

    numbers = list(range(3, threshold + 1, 2))
    root_of_threshold = threshold**0.5
    half = int((threshold + 1) / 2 - 1)
    idx = 0
    counter = 3
    while counter <= root_of_threshold:
        if numbers[idx]:
            idy = int((counter * counter - 3) / 2)
            numbers[idy] = 0
            while idy < half:
                numbers[idy] = 0
                idy += counter
        idx += 1
        counter = 2 * idx + 3
    return [2] + [number for number in numbers if number]
