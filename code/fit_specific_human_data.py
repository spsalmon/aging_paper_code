from gla_package.gla import siler_model_vectorized, classic_gla_model_vectorized, heligman_model_vectorized, aging_gompertz_makeham_vectorized
from gla_package.fit import fit_mortality_model, compute_fit_statistics
import pandas as pd
import numpy as np
import os

def fit_all_years(data_file, model, parameter_names, age_start, age_stop, initial_guess, bounds, fit_log=True):
    data = pd.read_csv(data_file)
    years = data["Year"].unique()
    results = []
    for year in years:
        year_data = data[data["Year"] == year]
        fit_age_stop = age_stop
        if year_data['Age'].max() < age_stop:
            fit_age_stop = year_data['Age'].max()
        if year_data['Age'].min() > age_start:
            print(f"Data for {os.path.basename(data_file)} and year {year} does not start at age {age_start}, skipping")
            continue
        try:
            parameters = fit_mortality_model(year_data, model, age_start, fit_age_stop, initial_guess, bounds, fit_log)
        except TypeError:
            print(f"Failed to fit {os.path.basename(data_file)} for year {year}")
            continue
        r2, rmse, AIC = compute_fit_statistics(year_data, model, parameters, age_start, age_stop, True)
        try:
            dict = {parameter_names[i]: parameters[i] for i in range(len(parameters))}
            dict["Year"] = year
            dict["R2"] = r2
            dict["RMSE"] = rmse
            dict["AIC"] = AIC
            dict["Country"] = os.path.splitext(os.path.basename(data_file))[0]
            results.append(dict)
        except IndexError:
            print(f"Parameter names is the wrong length, check that it matches the number of parameters in the model : len parameter names: {len(parameter_names)}, len parameters: {len(parameters)}")
            break
    return results

def fit_list_of_years(data_file, list_years, model, parameter_names, age_start, age_stop, initial_guess, bounds, fit_log=True):
    data = pd.read_csv(data_file)
    years = data["Year"].unique()
    list_years = [y for y in list_years if y in years]
    results = []
    for year in list_years:
        year_data = data[data["Year"] == year]
        fit_age_stop = age_stop
        if year_data['Age'].max() < age_stop:
            fit_age_stop = year_data['Age'].max()
        if year_data['Age'].min() > age_start:
            print(f"Data for {os.path.basename(data_file)} and year {year} does not start at age {age_start}, skipping")
            continue
        try:
            parameters = fit_mortality_model(year_data, model, age_start, fit_age_stop, initial_guess, bounds, fit_log)
        except TypeError:
            print(f"Failed to fit {os.path.basename(data_file)} for year {year}")
            continue
        r2, rmse, AIC = compute_fit_statistics(year_data, model, parameters, age_start, age_stop, True)
        try:
            dict = {parameter_names[i]: parameters[i] for i in range(len(parameters))}
            dict["Year"] = year
            dict["R2"] = r2
            dict["RMSE"] = rmse
            dict["AIC"] = AIC
            dict["Country"] = os.path.splitext(os.path.basename(data_file))[0]
            results.append(dict)
        except IndexError:
            print(f"Parameter names is the wrong length, check that it matches the number of parameters in the model : len parameter names: {len(parameter_names)}, len parameters: {len(parameters)}")
            break
    return results

data_dir = "/home/spsalmon/better_paper_code/mortality_data/human_mortality_data/population/life_table_both/"
output_dir = "/home/spsalmon/better_paper_code/parameters/human_parameters/population/65yo/"
os.makedirs(output_dir, exist_ok=True)

model_name = "GLA"

# assert model_name in ["siler", "heligman", "GLA"], "model must be one of siler, heligman, or GLA"

country = "GBR_NP"
list_years = [2009]

output_file = os.path.join(output_dir, f'{os.path.basename(os.path.normpath(data_dir))}_{model_name}_{country}.csv')

# list_of_countries = ["CAN", "FRATNP", "FRACNP", "ITA", "JPN", "NLD", "ESP", "USA", "GBR_NP"]


data_file = os.path.join(data_dir, f"{country}.csv")

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
    aging_guess = [0.009236162339816686, 0.042668562971162664, 0.0066301395123608015]
    # learning_guess = [0.003592451564100836, 40, 9.41817943e-02]
    learning_guess = [0.0003592451564100836, 40, 9.41817943e-02]
    # growth_guess = [1.00399978e-02, 9.23916941e-02]
    growth_guess = [0.05399978e-02, 1.23916941e-02]
    guess = np.concatenate((aging_guess, learning_guess, growth_guess))
    bounds = ((0, 0, 0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 80, 10, 1, 1))
    model = classic_gla_model_vectorized

elif model_name == "gompertz_makeham":
    parameter_names = ["a", "b", "c"]
    guess = [0.1, 0.1, 0.1]
    bounds = ((0, 0, 0), (10, 10, 10))
    model = aging_gompertz_makeham_vectorized
else:
    raise ValueError("model must be one of siler, heligman, or GLA")

output_df = pd.DataFrame()

print(f"fitting {os.path.basename(data_file)}")
results = fit_list_of_years(data_file, list_years, model, parameter_names, 0, 65, guess, bounds, True)
print(results)
output_df = pd.concat([output_df, pd.DataFrame(results)], ignore_index=True)

output_df.to_csv(output_file, index=False)