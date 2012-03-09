###########################
Scalefactors and intercepts
###########################

SPM Analyze and nifti1 images have *scalefactors*.  nifti1 images also have
*intercepts*.  If ``A`` is an array in memory, and ``S`` is the array that will
be written to disk, then::

    R = (A - intercept) / scalefactor

and ``R == S`` if ``R`` is already the data dtype we need to write.

If we load the image from disk, we exactly recover ``S`` (and ``R``).  To get
something approximating ``A`` (say ``Aprime``) we apply the intercept and
scalefactor::

    Aprime = (S * scalefactor) + intercept

In a perfect world ``A`` would be exactly the same as ``Aprime``.  However
``scalefactor`` and ``intercept`` are floating point values.  With floating
point, if ``r = (a - b) / c; p = (r * c) + b`` it is not necessarily true that
``p == a``. For example:

>>> import numpy as np
>>> a = 10
>>> b = np.e
>>> c = np.pi
>>> r = (a - b) / c
>>> p = (r * c) + b
>>> p == a
False

So there will be some error in this reconstruction, even when ``R`` is the same
type as ``S``.

More common is the situation where ``R`` is a different type from ``S``.  If
``R`` is of type ``r_dtype``, ``S`` is of type ``s_dtype`` and
``cast_function(R, dtype)`` is some function that casts ``R`` to the desired
type ``dtype``, then::

    R = (A - intercept) / scalefactor
    S = cast_function(R, s_dtype)
    R_prime = cast_function(S, r_dtype)
    A_prime = (R_prime * scalefactor) + intercept

The type of ``R`` will depend on what numpy did for upcasting ``A, intercept,
scalefactor``.

In order that ``cast_function(S, r_dtype)`` can best reverse ``cast_function(R,
s_dtype)``, the second needs to know the type of ``R``, which is not stored. The
type of ``R`` depends on the types of ``A`` and of ``intercept, scalefactor``.
We don't know the type of ``A`` because it is not stored.

``R`` is likely to be a floating point type because of the application of
scalefactor and intercept. If ``(intercept, scalefactor)`` are not the identity
(0, 1), then we can ensure that ``R`` is at minimum the type of the ``intercept,
scalefactor`` by making these be at least 1D arrays, so that floating point
types will upcast in ``R = (A - intercept) / scalefactor``.

The cast of ``R`` to ``S`` and back to ``R_prime`` can lose resolution if the
types of ``R`` and ``S`` have different resolution.

Our job is to select:

* scalefactor
* intercept
* ``cast_function``

such that we minimize some measure of difference between ``A`` and
``A_prime``.
