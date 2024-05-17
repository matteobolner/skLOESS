"""
=============================
Plotting LOESS results
=============================

An example plot of :class:`skLOESS.LOESS`
"""

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from itertools import product
from skLOESS import LOESS


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


degrees=[1,2,3]
smoothings=[0.3,0.5,0.7,1]

font = {'size'   : 20}

matplotlib.rc('font', **font)

num_rows = len(degrees)
num_cols = len(smoothings)
fig, axs = plt.subplots(num_rows, num_cols, figsize=(30, 20))



# Generate x values
# Loop through all parameter combinations and create a plot for each
for i, (degree, smoothing) in enumerate(product(degrees, smoothings)):
    row = i // num_cols
    col = i % num_cols
    estimator = LOESS(degree, smoothing)
    estimator.fit(X, y)
    predicted = estimator.predict(X)
    axs[row, col].scatter(X, y, label="Original Data")
    axs[row, col].plot(X, predicted , label="Predicted Data")
    axs[row, col].title.set_text(f"degree={degree}, smoothing={smoothing}")
    axs[row, col].legend(loc="best")
    axs[row, col].grid(True)
    axs[row, col].set_xlabel("x")
    axs[row, col].set_ylabel("y")

fig.suptitle("Plots of original and transformed data")

# Adjust layout

#fig.text(0.5, 0.04, 'x', ha='center', va='center')
#fig.text(0.06, 0.5, 'y', ha='center', va='center', rotation='vertical')
plt.tight_layout()
# Show plot
plt.show()
