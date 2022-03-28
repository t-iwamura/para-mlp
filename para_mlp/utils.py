import numpy as np
from numpy.typing import NDArray


def rmse(y_predict: NDArray, y_target: NDArray) -> float:
    return np.sqrt(np.mean(np.square(y_predict - y_target)))
