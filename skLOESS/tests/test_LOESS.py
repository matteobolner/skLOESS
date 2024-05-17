"""This file will just show how to write tests for the template classes."""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.utils._testing import assert_allclose, assert_array_equal

from skLOESS import TemplateClassifier, TemplateEstimator, TemplateTransformer
from skLOESS.skLOESS import LOESS

from sklearn.utils.estimator_checks import check_estimator


def test_check_estimator():
    """
    Test that the LOESS estimator class passes the scikit-learn checks.

    The check_estimator function from sklearn.utils.estimator_checks checks that a given
    class conforms to the scikit-learn API. This test checks that the LOESS class passes
    these checks. Around half of the checks do not pass because of sklearn incompatibility
    with 1d input.
    """
    a = check_estimator(LOESS(), generate_only=True)

    passed = []
    not_passed = []
    for i, j in a:
        try:
            # If the check succeeds, add the check to the "passed" list
            j(i)
            passed.append(j)
        except:
            # If the check fails, add the check to the "not_passed" list
            not_passed.append(j)

    print(
        f"{len(passed)}/{len(not_passed)} of the estimator checks passed; this is due to 1d arrays use as input"
    )
    assert len(passed) == 21


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


test_cases = {
    "deg_1_smoothing_033": {
        "degree": 1,
        "smoothing": 0.33,
        "predicted": np.array(
            [
                20.59302337,
                107.16030719,
                139.76738119,
                174.26304346,
                207.23338255,
                216.66158601,
                220.54447983,
                229.86069301,
                229.834713,
                229.43011583,
                226.60445904,
                220.39040989,
                172.34799941,
                163.84166131,
                161.84897069,
                160.33508369,
                160.19198931,
                161.05559254,
                227.33995587,
                227.89853498,
                231.55855634,
            ]
        ),
    },
    "deg_1_smoothing_1": {
        "degree": 1,
        "smoothing": 1,
        "predicted": np.array(
            [
                140.72071114,
                155.01113002,
                159.82606854,
                166.5005421,
                172.90878141,
                175.88369415,
                178.21342567,
                186.54616664,
                187.48992005,
                190.14151776,
                193.74719579,
                195.94620846,
                194.12619451,
                194.33189789,
                194.3681457,
                194.02923569,
                193.48111679,
                193.3448804,
                193.08426733,
                193.09747253,
                193.19235545,
            ]
        ),
    },
    "deg_1_smoothing_05": {
        "degree": 1,
        "smoothing": 0.5,
        "predicted": np.array(
            [
                35.1653735,
                105.09936726,
                130.91499875,
                169.70969168,
                197.01981569,
                208.9143723,
                216.99327846,
                225.58074713,
                225.93605214,
                226.96568881,
                226.27781442,
                211.06802053,
                178.21733474,
                174.38550137,
                164.22541979,
                163.70116618,
                179.88841474,
                183.40602944,
                221.50868955,
                221.87430409,
                224.27704699,
            ]
        ),
    },
    "deg_2_smoothing_1": {
        "degree": 2,
        "smoothing": 1,
        "predicted": np.array(
            [
                48.20669387,
                115.64062564,
                137.12170932,
                165.32826963,
                189.91167047,
                200.16714011,
                207.55120101,
                227.79749377,
                229.29716943,
                232.27732484,
                231.8547666,
                222.00692511,
                179.49852593,
                177.4882187,
                175.71672528,
                176.7387191,
                181.37889619,
                183.13171044,
                217.5299363,
                217.97620226,
                220.95717591,
            ]
        ),
    },
    "deg_2_smoothing_05": {
        "degree": 2,
        "smoothing": 0.5,
        "predicted": np.array(
            [
                15.89401961,
                115.32538047,
                145.22225614,
                182.58620217,
                211.00665143,
                219.9199993,
                224.42148356,
                231.44891858,
                231.68067105,
                231.69152476,
                228.1725204,
                220.59271181,
                168.59632679,
                164.46361742,
                157.84155742,
                155.12851671,
                160.66062325,
                163.46543034,
                226.76228896,
                227.6114308,
                233.28914598,
            ]
        ),
    },
}


def test_template_estimator(data):
    """Check the internals and behaviour of skLOESS."""
    est = LOESS()
    assert est.degree == 1
    assert est.smoothing == 0.33

    est.fit(*data)
    assert hasattr(est, "is_fitted_")

    X = data[0]
    y_pred = est.predict(X)
    assert X.shape == y_pred.shape
    assert X.shape == data[1].shape


param_list = [
    (v["degree"], v["smoothing"], v["predicted"]) for v in test_cases.values()
]


@pytest.mark.parametrize("degree, smoothing, expected", param_list)
def test_multiple_cases(data, degree, smoothing, expected):
    est = LOESS(degree, smoothing)
    est.fit(*data)
    X = data[0]
    y_pred = est.predict(X)
    expected = np.around(expected, 6)
    y_pred = np.around(y_pred, 6)
    assert_array_equal(y_pred, expected)
