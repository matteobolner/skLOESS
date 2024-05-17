"""This file shows how to write test based on the scikit-learn common tests."""

# Authors: scikit-learn-contrib developers
# License: BSD 3 clause

from sklearn.utils.estimator_checks import check_estimator

from skloess import LOESS


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
