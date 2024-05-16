import numpy as np
import math
from sklearn.base import BaseEstimator, RegressorMixin, _fit_context
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.estimator_checks import check_estimator


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


class LOESS(RegressorMixin, BaseEstimator):
    """

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstration of how to pass and store parameters.

    Attributes
    ----------
    is_fitted_ : bool
        A boolean indicating whether the estimator has been fitted.
    use_matrix : bool
    n_neighbors_


    Examples
    --------
    >>> from skltemplate import TemplateEstimator
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> estimator = TemplateEstimator()
    >>> estimator.fit(X, y)
    TemplateEstimator()
    """

    # This is a dictionary allowing to define the type of parameters.
    # It used to validate parameter within the `_fit_context` decorator.
    _parameter_constraints = {
        "degree": [int],
        "smoothing": [float],
        "use_matrix": [bool],
    }

    def __init__(self, degree=1, smoothing=0.33, use_matrix=False):
        self.degree = degree
        self.smoothing = smoothing
        self.use_matrix = use_matrix

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        # `_validate_data` is defined in the `BaseEstimator` class.
        # It allows to:
        # - run different checks on the input data;
        # - define some attributes associated to the input data: `n_features_in_` and
        #   `feature_names_in_`.
        X, y = self._validate_data(X, y, accept_sparse=True)
        X = X.flatten()
        self.norm_X_global_, self.min_X_global, self.max_X_global = normalize_array(X)
        self.norm_y_global_, self.min_y_global, self.max_y_global = normalize_array(y)
        self.n_neighbors_ = round(self.smoothing * X.shape[0])
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def estimate(self, X):

        norm_X_local = normalize_value(X, self.min_X_global, self.max_X_global)
        distances = np.abs(self.norm_X_global_ - norm_X_local)
        min_range = get_min_range(distances, self.n_neighbors_)
        weights = get_weights(distances, min_range)

        if self.use_matrix:
            y = estimate_polynomial(
                self.n_neighbors_,
                weights,
                self.degree,
                self.norm_X_global_,
                self.norm_y_global_,
                norm_X_local,
                min_range,
            )
        else:
            y = estimate_linear(
                min_range,
                self.norm_X_global_,
                self.norm_y_global_,
                weights,
                norm_X_local,
            )
        denormalized_y = denormalize(y, self.min_y_global, self.max_y_global)
        return denormalized_y

    def predict(self, X):
        """A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        # Check if fit had been called
        check_is_fitted(self)
        # We need to set reset=False because we don't want to overwrite `n_features_in_`
        # `feature_names_in_` but only check that the shape is consistent.
        X = self._validate_data(X, accept_sparse=True, reset=False)
        X = X.flatten()
        predicted = []
        for i in X:
            predicted.append(self.estimate(i))
        return np.array(predicted)
