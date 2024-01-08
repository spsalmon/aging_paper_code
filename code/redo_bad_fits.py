from gla_package.gla import siler_model_vectorized, classic_gla_model_vectorized, heligman_model_vectorized, aging_gompertz_makeham_vectorized
from gla_package.fit import fit_mortality_model, compute_fit_statistics
from scipy.signal import find_peaks
import pandas as pd
import numpy as np
import os

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

def refit_bad_years(param_file, country_params, model, parameter_names, age_start, age_stop, initial_guess, bounds, fit_log=True):
    years = country_params["Year"].to_numpy()
    aic = country_params["AIC"].to_numpy()
    param_df = pd.read_csv(param_file)
    
    # find peaks in the AIC
    peaks, _ = find_peaks(aic, prominence=5)
    years_with_peaks = years[peaks]
    aic_peaks = aic[peaks]

    # fit the years with peaks
    country = country_params["Country"].iloc[0]
    country_data_file = os.path.join(data_dir, f"{country}.csv")
    country_data = pd.read_csv(country_data_file)
    results = fit_list_of_years(country_data_file, years_with_peaks, model, parameter_names, age_start, age_stop, initial_guess, bounds, fit_log=True)

    # print aic improvement
    for i in range(len(years_with_peaks)):
        # if the new AIC is signficantly better, print the improvement
        if results[i]["AIC"] < 1.02*aic_peaks[i]:
            print(f"{years_with_peaks[i]}: {aic_peaks[i]} -> {results[i]['AIC']}")
            # replace the old parameters with the new ones
            param_df.loc[(param_df["Country"] == country) & (param_df["Year"] == years_with_peaks[i]), parameter_names] = [results[i][p] for p in parameter_names]
            r2, rmse, AIC = compute_fit_statistics(country_data[country_data["Year"] == years_with_peaks[i]], model, np.array([results[i][p] for p in parameter_names]), age_start, age_stop, True)
            param_df.loc[(param_df["Country"] == country) & (param_df["Year"] == years_with_peaks[i]), ["R2", "RMSE", "AIC"]] = [r2, rmse, AIC]
    
    param_df.to_csv(output_file, index=False)

data_dir = "/home/spsalmon/better_paper_code/mortality_data/human_mortality_data/population/life_table_both/"
output_dir = "/home/spsalmon/better_paper_code/parameters/human_parameters/cohort/65yo/"
os.makedirs(output_dir, exist_ok=True)

model_name = "GLA"

# assert model_name in ["siler", "heligman", "GLA"], "model must be one of siler, heligman, or GLA"

output_file = os.path.join(output_dir, f'{os.path.basename(os.path.normpath(data_dir))}_{model_name}.csv')

list_of_countries = ["CAN", "FRATNP", "FRACNP", "ITA", "JPN", "NLD", "ESP", "USA", "GBR_NP"]

# load data
data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".csv")]
# param_file = os.path.join(output_dir, f'{os.path.basename(os.path.normpath(data_dir))}_{model_name}.csv')
param_file = "/home/spsalmon/better_paper_code/parameters/human_parameters/population/65yo/life_table_both_GLA_refit.csv"
output_file = "/home/spsalmon/better_paper_code/parameters/human_parameters/population/65yo/life_table_both_GLA_refit.csv"
param_df = pd.read_csv(param_file)

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
    # # learning_guess = [0.003592451564100836, 40, 9.41817943e-02]
    # learning_guess = [0.010592451564100836, 40, 9.41817943e-02]
    # growth_guess = [1.00399978e-02, 9.23916941e-02]

    # learning_guess = [0.003592451564100836, 40, 9.41817943e-02]
    learning_guess = [0.0003592451564100836, 40, 9.41817943e-02]
    # growth_guess = [1.00399978e-02, 9.23916941e-02]
    growth_guess = [0.10399978e-02, 1.23916941e-02]
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
for country in list_of_countries:
    country_data = param_df[param_df["Country"] == country]
    country_data = country_data[country_data["Year"] > 1948]
    print(f"fitting {os.path.basename(country)}")
    refit_bad_years(param_file, country_data, model, parameter_names, 0, 65, guess, bounds, True)
    # print(f"fitting {os.path.basename(file)}")
    # results = fit_all_years(file, model, parameter_names, 0, 65, guess, bounds, True)
    # output_df = pd.concat([output_df, pd.DataFrame(results)], ignore_index=True)

# output_df.to_csv(output_file, index=False)