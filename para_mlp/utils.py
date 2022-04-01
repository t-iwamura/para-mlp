import numpy as np
from numpy.typing import NDArray


def rmse(y_predict: NDArray, y_target: NDArray) -> float:
    """Calculate RMSE of objective variable

    Args:
        y_predict (NDArray): predicted objective variable
        y_target (NDArray): target objected variable

    Returns:
        float: RMSE
    """
    return np.sqrt(np.mean(np.square(y_predict - y_target)))
