import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize, least_squares
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

def fit_mortality_model(age_qx_dataframe, model, age_start, age_stop, initial_guess, bounds, fit_log=False):
    """
    Fit a mortality model to the data.
    
    Parameters
    ----------
    age : numpy.array
        The ages of the mortality data.
    qx : numpy.array
        The mortality rates.
    model : function
        The mortality model to fit.
    age_start : int
        The age to start fitting the model.
    age_stop : int
        The age to stop fitting the model.
    initial_guess : numpy.array
        The initial guess for the parameters of the model.
    bounds : tuple
        The bounds for the parameters of the model.
    """

    age_qx_dataframe = age_qx_dataframe[(age_qx_dataframe["Age"] >= age_start) & (age_qx_dataframe["Age"] <= age_stop)]
    age_array = (age_qx_dataframe["Age"].values).astype(float)
    qx_array = age_qx_dataframe["qx"].values

    def cost_function(params):
        if fit_log:
            return np.log10(np.asarray(model(age_array, *params))+1e-10) - np.log10(qx_array+1e-10)
        else:
            return model(age_array, *params) - qx_array
    
    res = least_squares(cost_function, initial_guess, bounds=bounds, jac="3-point", loss="soft_l1")

    return res.x

def rss(modeled_qx, qx):
    return norm(modeled_qx - qx)

def log_likelihood(age_array, parameters, rss):
    n = age_array.shape[0]
    k = parameters.shape[0]

    ll = -(n * 1/2) * (1 + np.log(2 * np.pi)) - (n / 2) * np.log((rss**2)/ n)

    return ll

def aic(age_array, parameters, modeled_qx, qx):
    residuals = rss(modeled_qx, qx)
    ll = log_likelihood(age_array, parameters, residuals)
    k = parameters.shape[0]
    n = age_array.shape[0]

    AIC = (-2 * ll) + (2 * k)

    return AIC


def compute_fit_statistics(age_qx_dataframe, model, parameters, age_start, age_stop, compare_log = True):
    age_qx_dataframe = age_qx_dataframe[(age_qx_dataframe["Age"] >= age_start) & (age_qx_dataframe["Age"] <= age_stop)]
    age_array = (age_qx_dataframe["Age"].values).astype(float)
    qx_array = age_qx_dataframe["qx"].values
    modeled_qx = model(age_array, *parameters).base

    if compare_log:
        qx_array = np.log10(qx_array+ 1e-10)
        modeled_qx = np.log10(modeled_qx + 1e-10)
        r2 = r2_score(qx_array, modeled_qx)
        rmse = mean_squared_error(qx_array, modeled_qx, squared=True)
        AIC = aic(age_array, parameters, modeled_qx, qx_array)
    else:
        r2 = r2_score(qx_array, modeled_qx)
        rmse = mean_squared_error(qx_array, modeled_qx, squared=True)
        AIC = aic(age_array, parameters, modeled_qx, qx_array)
    return r2, rmse, AIC