# skloess #
  
[![PyPI version](https://badge.fury.io/py/skloess.svg)](https://badge.fury.io/py/skloess)

scikit-learn compatible implementation of the 
 [Locally Estimated Scatterplot Smoothing (LOESS) algorithm ](https://en.wikipedia.org/wiki/Local_regression).

The code for the python version of loess is based on https://github.com/joaofig/pyloess

References:  
- https://en.wikipedia.org/wiki/Local_regression
- https://www.itl.nist.gov/div898/handbook/pmd/section1/dep/dep144.htm
- https://www.itl.nist.gov/div898/handbook/pmd/section4/pmd423.htm
- https://towardsdatascience.com/loess-373d43b03564


## How to install ##

Install with `pip`:

```shell
pip install skloess
```

## Dependencies ##

* numpy
* math
* scikit-learn
* matplotlib
* pytest for running the test suite

## How to use ##

Download, import and do as you would with any other scikit-learn method:
* fit(X, y)
* predict(X)  
  
Keep in mind that the model requires 1-dimensional input, i.e. a list/array/pandas series for both X and y. X and y must have the same length.


## Description ##
An excellent explanation of the pyloess code can be found at https://towardsdatascience.com/loess-373d43b03564.
This package simply formats the pyloess code into a sklearn estimator.

## Parameters ##

__estimator__ : object
   > A supervised learning estimator, with a 'fit' and 'predict' method.

__degree__ : int, default = 1
   > Degree of the polynomial. Default value of 1 is a linear implementation.

__smoothing__ : float, default = 0.33
   > Smoothing value. This value is used to determine the number of closest points to use for the fitting and estimation process. For example, a value of 0.33 over 21 X values means that 7 closest points will be chosen. The smoothing parameter is a number between (λ + 1) / n and 1, with λ denoting the degree of the local polynomial and n denoting the total number of observations.

![alt text](https://raw.githubusercontent.com/matteobolner/skLOESS/main/examples/loess_grid.png)

## Examples ##

```python
from skloess import LOESS

# load X and y

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


# define estimator with default params
est = LOESS(degree=1, smoothing=0.33)

#fit the estimator
est.fit(X, y)

# predict the y values for the original X values
predicted=est.predict(X)

```
