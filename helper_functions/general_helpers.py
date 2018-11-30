from scipy import stats
import numpy as np

def calc_comparison(actual, predicted):
    """
    actual: array for the actual transfer integrals
    predicted: array for the predicted TIs
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        actual.flatten(), predicted.flatten()
    )
    mae = np.mean(abs(predicted-actual))
    return r_value, mae
