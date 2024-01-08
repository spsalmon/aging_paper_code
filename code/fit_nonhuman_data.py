from gla_package.gla import siler_model_vectorized, classic_gla_model_vectorized, heligman_model_vectorized
from gla_package.fit import fit_mortality_model, compute_fit_statistics
import pandas as pd
import numpy as np
import cma
from numpy.linalg import norm
import os

def fit_mortality_model_cma(age_qx_dataframe, model, age_start, age_stop, initial_guess, bounds, sigma = 0.05, fit_log=False):
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
    """

    age_qx_dataframe = age_qx_dataframe[(age_qx_dataframe["Age"] >= age_start) & (age_qx_dataframe["Age"] <= age_stop)]
    age_array = (age_qx_dataframe["Age"].values).astype(float)
    qx_array = age_qx_dataframe["qx"].values
    global f 
    def f(params):
        norm_under_bounds = (np.array(bounds[0]) - params)
        norm_under_bounds[norm_under_bounds < 0] = 0
        norm_under_bounds = np.exp(np.abs(norm_under_bounds)) - np.ones_like(norm_under_bounds)
        norm_under_bounds = norm(norm_under_bounds)

        norm_over_bounds = (np.array(bounds[1]) - params)
        norm_over_bounds[norm_over_bounds > 0] = 0
        norm_over_bounds = np.exp(np.abs(norm_over_bounds)) - np.ones_like(norm_over_bounds)
        norm_over_bounds = norm(norm_over_bounds)

        return norm(np.log10(np.asarray(model(age_array, *params))+1e-10) - np.log10(qx_array+1e-10)) + 10*norm_under_bounds + 10*norm_over_bounds

    es = cma.CMAEvolutionStrategy(initial_guess, sigma)
    es.optimize(f, iterations=5000, n_jobs=-1)

    return es.result.xbest

def fit_species(data_file, model, model_name, parameter_names, initial_guess, bounds, fit_log=True):
    data = pd.read_csv(data_file)
    
    age_start = 0
    age_stop = data['Age'].max()

    if model_name == "GLA":
        guess[4] = age_stop/2
        bounds = ((0, 0, 0, 0, 0, 0, 0, 0), (1, 1, 1, 1, age_stop, 10, 1, 1))
    if model_name == "heligman":
        guess[4] = age_stop/2
        guess[5] = age_stop/2
        bounds = ((0, 0, 0, 0, 0, 0, 0, 0), (10, 10, 10, 10, age_stop, age_stop, 10, 10))


    try:
        parameters_least_squares = fit_mortality_model(data, model, age_start, age_stop, initial_guess, bounds, fit_log)
        parameters_cma = fit_mortality_model_cma(data, model, age_start, age_stop, initial_guess, bounds, 0.05, fit_log)

        rmse_least_squares = compute_fit_statistics(data, model, parameters_least_squares, age_start, age_stop, True)[1]
        rmse_cma = compute_fit_statistics(data, model, parameters_cma, age_start, age_stop, True)[1]

        if rmse_least_squares < rmse_cma or np.any(bounds[0] -parameters_cma > 0) or np.any(bounds[1] - parameters_cma < 0):
            parameters = parameters_least_squares
        else:
            parameters = parameters_cma
    except TypeError:
        print(f"Failed to fit {os.path.basename(data_file)}")
        return
    r2, rmse, AIC = compute_fit_statistics(data, model, parameters, age_start, age_stop, True)
    try:
        dict = {parameter_names[i]: parameters[i] for i in range(len(parameters))}
        dict["Subphylum"] = data["subphylum"].unique()[0]
        dict["R2"] = r2
        dict["RMSE"] = rmse
        dict["AIC"] = AIC
        dict["Species"] = os.path.splitext(os.path.basename(data_file))[0]
        results = [dict]
    except IndexError:
        print(f"Parameter names is the wrong length, check that it matches the number of parameters in the model : len parameter names: {len(parameter_names)}, len parameters: {len(parameters)}")
        return
    return results

data_dir = "/home/spsalmon/better_paper_code/mortality_data/animals_and_plants/data/"
output_dir = "/home/spsalmon/better_paper_code/parameters/animals_and_plants/"
os.makedirs(output_dir, exist_ok=True)

model_name = "GLA"

# assert model_name in ["siler", "heligman", "GLA"], "model must be one of siler, heligman, or GLA"

output_file = os.path.join(output_dir, f'animals_and_plants_{model_name}.csv')

# load data
data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".csv")]

if model_name == "siler":
    parameter_names = ["a1", "a2", "a3", "b1", "b3"]
    guess = [0.1, 0.1, 0.1, 3, 0.1]
    bounds = ((0, 0, 0, 0, 0), (10, 10, 10, 10, 10))
    model = siler_model_vectorized

elif model_name == "heligman":
    parameter_names = ["A", "B", "C", "D", "E", "F", "G", "H"]
    guess = [0.00184, 0.0189, 0.1189, 0.00110, 13.55, 20.43, 0.0000711, 1.0992]
    bounds = ((0, 0, 0, 0, 0, 0, 0, 0), (10, 10, 10, 10, 80, 80, 10, 10))
    model = heligman_model_vectorized

elif model_name == "GLA":
    parameter_names = ["a", "b", "c", "Lmax", "k_learning", "n", "Gmax", "growth_rate"]
    aging_guess = [0.009236162339816686, 0.0042668562971162664, 0.0066301395123608015]
    learning_guess = [0.0003592451564100836, 40, 9.41817943e-02]
    growth_guess = [1.00399978e-02, 9.23916941e-02]
    guess = np.concatenate((aging_guess, learning_guess, growth_guess))
    bounds = ((0, 0, 0, 0, 0, 0, 0, 0), (2, 2, 2, 2, 80, 10, 2, 2))
    model = classic_gla_model_vectorized

else:
    raise ValueError("model must be one of siler, heligman, or GLA")

output_df = pd.DataFrame()
for file in data_files:
    print(f"fitting {os.path.basename(file)}")
    results = fit_species(file, model, model_name, parameter_names, guess, bounds, True)
    output_df = pd.concat([output_df, pd.DataFrame(results)], ignore_index=True)

output_df.to_csv(output_file, index=False)