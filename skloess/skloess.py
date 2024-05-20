"""
skloess: An extension of scikit-learn for Locally Estimated Scatterplot Smoothing
=====================================================================

This module extends the scikit-learn library by providing estimators for Locally Estimated Scatterplot Smoothing (LOESS).
LOESS is a technique for fitting a smooth curve to noisy data.

This implementation is based on the pyloess code (see https://github.com/joaofig/pyloess for details).

See also:
- https://en.wikipedia.org/wiki/Local_regression
- https://www.itl.nist.gov/div898/handbook/pmd/section1/dep/dep144.htm
- https://www.itl.nist.gov/div898/handbook/pmd/section4/pmd423.htm

"""

import numpy as np
import math
from sklearn.base import BaseEstimator, RegressorMixin, _fit_context
from sklearn.utils.validation import check_is_fitted


def tricubic(x):
    """
    Tri-cubic weighting function.

    Parameters
    ----------
    x : np.array
        Input array for which the weights will be computed

    Returns
    -------
    np.array
        Values of the weights.

    """
    y = np.zeros_like(x)

    idx = (x >= -1) & (x <= 1)

    y[idx] = np.power(1.0 - np.power(np.abs(x[idx]), 3), 3)

    return y


def normalize_value(value, min_value, max_value):
    """
    Normalize a value to the range [0, 1].

    Parameters
    ----------
    value : float
        The value to be normalized.
    min_value : float
        The minimum value in the range.
    max_value : float
        The maximum value in the range.

    Returns
    -------
    float
        The normalized value.
    """
    # Calculate the normalized value by subtracting the minimum value
    # and dividing by the range between the minimum and maximum values
    return (value - min_value) / (max_value - min_value)


def normalize_array(array):
    """
    Normalize the input array by subtracting the minimum value and dividing by the range.

    Parameters
    ----------
    array : np.array
        Input array to be normalized.

    Returns
    -------
    np.array, float, float
        Normalized array, minimum value, and maximum value.
    """
    # Compute the minimum value of the array
    array = array.flatten()
    min_val = np.min(array)

    # Compute the maximum value of the array
    max_val = np.max(array)

    # Normalize the array by subtracting the minimum value and dividing by the range
    # norm_array = (array - min_val) / (max_val - min_val)
    norm_array = np.vectorize(normalize_value)(array, min_val, max_val)
    # Return the normalized array, minimum value, and maximum value
    return norm_array, min_val, max_val


def get_min_range(distances, n):
    """
    Find the indices of the `n` closest points to the minimum distance.

    Parameters
    ----------
    distances : np.array
        Array of distances.
    n : int
        Number of closest points to return.

    Returns
    -------
    np.array
        Indices of the closest points.
    """
    # Find the minimum distance
    min_idx = np.argmin(distances)
    # Get the length of the array
    n_items = len(distances)

    # Check if the minimum distance is on the boundary
    if min_idx == 0:
        # If the minimum distance is at the beginning of the array,
        # return the first `n` elements
        return np.arange(0, n, dtype=np.int64)
    if min_idx == n_items - 1:
        # If the minimum distance is at the end of the array, return the last `n` elements
        return np.arange(n_items - n, n_items, dtype=np.int64)

    # Initialize the array of indices
    min_range = [min_idx]
    # Iterate until the size of the array is `n`
    while len(min_range) < n:
        # Get the first and last elements of the array
        i0 = min_range[0]
        i1 = min_range[-1]

        # Check if the first element is at the beginning of the array
        if i0 == 0:
            # If it is, add the next element to the array
            min_range.append(i1 + 1)
        # Check if the last element is at the end of the array
        elif i1 == n_items - 1:
            # If it is, add the previous element to the beginning of the array
            min_range.insert(0, i0 - 1)
        # Check if the distance to the left is smaller than the distance to the right
        elif distances[i0 - 1] < distances[i1 + 1]:
            # If it is, add the previous element to the beginning of the array
            min_range.insert(0, i0 - 1)
        # Otherwise, add the next element to the array
        else:
            min_range.append(i1 + 1)
    # Return the array of indices

    return np.array(min_range, dtype=np.int64)


def get_weights(distances, min_range):
    """
    Calculate weights for each point in the given range.

    Parameters
    ----------
    distances : np.array
        Array of distances.
    min_range : np.array
        Array of indices of the points within the minimal-distance window.

    Returns
    -------
    np.array
        Array of weights.
    """
    # Find the maximum distance within the range
    local_distances = distances[min_range]
    max_local_distance = np.max(local_distances)

    # Normalize the distances within the range
    normalized_local_distances = local_distances / max_local_distance

    # Apply tricubic function to normalized distances
    weights = tricubic(normalized_local_distances)

    return weights


def denormalize(value, min_value, max_value):
    """
    Denormalize a value from the range [0, 1] to the original range.

    Parameters
    ----------
    value : float
        The normalized value to be denormalized.
    min_value : float
        The minimum value in the original range.
    max_value : float
        The maximum value in the original range.

    Returns
    -------
    float
        The denormalized value.
    """
    # Multiply the normalized value by the range and add the minimum value
    # to get the denormalized value
    return value * (max_value - min_value) + min_value


def estimate_polynomial(
    n_neighbors,
    weights,
    degree,
    norm_X_global,
    norm_y_global,
    norm_X_local,
    min_range,
):
    wm = np.multiply(np.eye(n_neighbors), weights)
    xm = np.ones((n_neighbors, degree + 1))

    xp = np.array([[math.pow(norm_X_local, p)] for p in range(degree + 1)])
    for i in range(1, degree + 1):
        xm[:, i] = np.power(norm_X_global[min_range], i)

    ym = norm_y_global[min_range]
    xmt_wm = np.transpose(xm) @ wm
    beta = np.linalg.pinv(xmt_wm @ xm) @ xmt_wm @ ym
    y = (beta @ xp)[0]
    return y


def estimate_linear(min_range, norm_X_global, norm_y_global, weights, norm_X_local):
    xx = norm_X_global[min_range]
    yy = norm_y_global[min_range]
    sum_weight = np.sum(weights)
    sum_weight_x = np.dot(xx, weights)
    sum_weight_y = np.dot(yy, weights)
    sum_weight_x2 = np.dot(np.multiply(xx, xx), weights)
    sum_weight_xy = np.dot(np.multiply(xx, yy), weights)

    mean_x = sum_weight_x / sum_weight
    mean_y = sum_weight_y / sum_weight

    b = (sum_weight_xy - mean_x * mean_y * sum_weight) / (
        sum_weight_x2 - mean_x * mean_x * sum_weight
    )
    a = mean_y - b * mean_x
    y = a + b * norm_X_local
    return y


def validate_smoothing(smoothing, degree, n_obs):
    """
    Validate smoothing parameter for LOESS

    Parameters
    ----------
    smoothing : float
        Smoothing parameter
    degree : int
        Degree of the polynomial used to estimate the data
    n_obs : int
        Number of observations

    Returns
    -------
    smoothing : float
        Validated smoothing parameter

    Raises
    ------
    ValueError
        If smoothing is not between (degree + 1) / n_obs and 1
    """
    smoothing_min = (degree + 1) / n_obs
    if smoothing_min <= smoothing <= 1:
        return smoothing
    raise ValueError(
        f"Smoothing for a polynomial with {degree} degrees \
        must be between {smoothing_min} and 1".format(
            degree=degree, smoothing_min=smoothing_min
        )
    )


def to_numpy_array(data):
    """
    Converts input data into a NumPy array.

    Parameters:
    data (array-like): Input data, which can be a list, NumPy array, pandas Series, etc.

    Returns:
    np.ndarray: Converted NumPy array.
    """
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            return data.reshape(-1, 1)
        elif data.ndim >= 1:
            if data.shape[0] > 1:
                raise ValueError("Input numpy arrays must be 1 x n shaped")
            else:
                return data
    elif isinstance(data, (list, tuple)):
        return np.array(data).reshape(-1, 1)
    elif isinstance(data, pd.Series):
        return data.values.reshape(-1, 1)
    else:
        raise TypeError(f"Unsupported input type: {type(data)}")


class LOESS(RegressorMixin, BaseEstimator):
    """

    Parameters
    ----------
    degree : int, default=1
        Degree of the polynomial used to estimate the data.
    smoothing : float, default=0.33
        The amount of smoothing to apply to the data.
        This determines how many of the closest points will be used in the estimation.
    Attributes
    ----------
    is_fitted_ : bool
        A boolean indicating whether the estimator has been fitted.
    n_neighbors_: int
        Number of closest points.

    Examples
    --------
    >>> from skloess import LOESS
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.random.rand((100, ))
    >>> estimator = LOESS()
    >>> estimator.fit(X, y)
    >>> estimator.predict(X)
    """

    _parameter_constraints = {
        "degree": [int],
        "smoothing": [float, int],
    }

    def __init__(self, degree=1, smoothing=0.33):
        self.degree = degree
        self.smoothing = smoothing

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X : {array-like}, shape (n, 1)
            The X coordinates of the input. Must be 1-dimensional (list/array/pandas series)

        y : {array-like}, shape (n, 1)
            The y coordinates of the input. Must be 1-dimensional (list/array/pandas series)

        Returns
        -------
        self : object
            Returns self.
        """
        X = to_numpy_array(X)
        validate_smoothing(self.smoothing, self.degree, len(X))
        self.n_neighbors_ = round(self.smoothing * X.shape[0])

        X, y = self._validate_data(X, y, accept_sparse=True, reset=True)
        self.norm_X_global_, self.min_X_global, self.max_X_global = normalize_array(X)
        self.norm_y_global_, self.min_y_global, self.max_y_global = normalize_array(y)
        self.is_fitted_ = True
        return self

    def estimate(self, X):
        if X == np.nan:
            return np.nan
        norm_X_local = normalize_value(X, self.min_X_global, self.max_X_global)
        distances = np.abs(self.norm_X_global_ - norm_X_local)
        min_range = get_min_range(distances, self.n_neighbors_)
        weights = get_weights(distances, min_range)
        if self.degree == 1:
            y = estimate_linear(
                min_range,
                self.norm_X_global_,
                self.norm_y_global_,
                weights,
                norm_X_local,
            )
        else:
            y = estimate_polynomial(
                self.n_neighbors_,
                weights,
                self.degree,
                self.norm_X_global_,
                self.norm_y_global_,
                norm_X_local,
                min_range,
            )

        denormalized_y = denormalize(y, self.min_y_global, self.max_y_global)
        return denormalized_y

    def predict(self, X):
        """Estimate the y values from a collection of X values.

        Parameters
        ----------
        X : {array-like}, shape (n, 1)
            The collection of values to estimate. Must be 1-dimensional (list/array/pandas series)

        Returns
        -------
        y : ndarray, shape (n, 1)
            Returns an array containing the estimated values.
        """
        # Check if fit had been called
        check_is_fitted(self)
        X = to_numpy_array(X)
        X = self._validate_data(X, accept_sparse=True, reset=False)
        predicted = np.vectorize(self.estimate)(X)
        predicted = predicted.flatten()
        return predicted
