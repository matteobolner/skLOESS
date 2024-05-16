"""This file will just show how to write tests for the template classes."""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.utils._testing import assert_allclose, assert_array_equal

from skLOESS import TemplateClassifier, TemplateEstimator, TemplateTransformer
from skLOESS.skLOESS import LOESS

# Authors: scikit-learn-contrib developers
# License: BSD 3 clause


@pytest.fixture
def data():
    X = np.array(
        [
            0.5578196,
            2.0217271,
            2.5773252,
            3.4140288,
            4.3014084,
            4.7448394,
            5.1073781,
            6.5411662,
            6.7216176,
            7.2600583,
            8.1335874,
            9.1224379,
            11.9296663,
            12.3797674,
            13.2728619,
            14.2767453,
            15.3731026,
            15.6476637,
            18.5605355,
            18.5866354,
            18.7572812,
        ]
    )
    y = np.array(
        [
            18.63654,
            103.49646,
            150.35391,
            190.51031,
            208.70115,
            213.71135,
            228.49353,
            233.55387,
            234.55054,
            223.89225,
            227.68339,
            223.91982,
            168.01999,
            164.95750,
            152.61107,
            160.78742,
            168.55567,
            152.42658,
            221.70702,
            222.69040,
            243.18828,
        ]
    )
    return (X, y)

deg_1_smoothing_033_no_matrix=np.array([ 20.59302337, 107.16030719, 139.76738119, 174.26304346,
       207.23338255, 216.66158601, 220.54447983, 229.86069301,
       229.834713  , 229.43011583, 226.60445904, 220.39040989,
       172.34799941, 163.84166131, 161.84897069, 160.33508369,
       160.19198931, 161.05559254, 227.33995587, 227.89853498,
       231.55855634])


def test_template_estimator(data):
    """Check the internals and behaviour of `TemplateEstimator`."""
    est = LOESS()
    assert est.degree == "1"
    assert est.use_matrix == False

    est.fit(*data)
    assert hasattr(est, "is_fitted_")

    X = data[0]
    y_pred = est.predict(X)
    assert_array_equal(y_pred, deg_1_smoothing_033_no_matrix)


def test_template_transformer(data):
    """Check the internals and behaviour of `TemplateTransformer`."""
    X, y = data
    trans = TemplateTransformer()
    assert trans.demo_param == "demo"

    trans.fit(X)
    assert trans.n_features_in_ == X.shape[1]

    X_trans = trans.transform(X)
    assert_allclose(X_trans, np.sqrt(X))

    X_trans = trans.fit_transform(X)
    assert_allclose(X_trans, np.sqrt(X))


def test_template_classifier(data):
    """Check the internals and behaviour of `TemplateClassifier`."""
    X, y = data
    clf = TemplateClassifier()
    assert clf.demo_param == "demo"

    clf.fit(X, y)
    assert hasattr(clf, "classes_")
    assert hasattr(clf, "X_")
    assert hasattr(clf, "y_")

    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)
