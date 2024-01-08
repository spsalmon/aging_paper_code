from time import perf_counter
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.integrate
import scipy.optimize
from gla_package.gla import aging_gompertz_makeham, learning_function, growth_function, fertility_brass_polynomial
from gla_package.landscape import compute_fitness_grid, compute_fitness_grid_point
from joblib import Parallel, delayed

def generate_grid(grid_type, b_min, b_max, lmax_min, lmax_max, dim = 5):
	grid = np.zeros((dim*dim, 2))

	if grid_type == "regular":
		b_grid = np.linspace(b_min, b_max, dim)
		Lmax_grid = np.linspace(lmax_min, lmax_max, dim)
		B, L = np.meshgrid(b_grid, Lmax_grid)

		grid[:, 0] = B.flatten()
		grid[:, 1] = L.flatten()
	return grid

def compute_reduced_b_grid(grid, b_decrease):
	reduced_b_grid = grid.copy()
	reduced_b_grid[:, 0] = reduced_b_grid[:, 0]*b_decrease
	return reduced_b_grid

def parallel_compute_fitness_grid(grid, aging_function, learning_function, growth_function, fertility_function, aging_args, learning_args, growth_args, fertility_args):
	fitness = Parallel(n_jobs=4)(delayed(compute_fitness_grid_point)(point, aging_function, learning_function, growth_function, fertility_function, aging_args, learning_args, growth_args, fertility_args) for point in grid)
	return fitness

	
aging_args = [0.00275961297460256,0.04326224872667336,0.025201676835511704]  # Arguments for aging_gompertz
learning_args = [0.01606792505529796,39.006865144958745,0.11060749334680318]  # Arguments for learning_function
growth_args = [0.05168141300917714,0.08765165352033985]  # Arguments for growth_function
fertility_args = [2.455e-5, 14.8, 32.836]  # Arguments for fertility_function


columns = ["b", "lmax", "fitness_difference"]
grid = generate_grid("regular", 1e-5, 0.2, 0.0, 0.125, dim = 2)
reduced_b_grid = compute_reduced_b_grid(grid, 0.98)

t0 = perf_counter()
fitness_base_b = parallel_compute_fitness_grid(grid, aging_gompertz_makeham, learning_function, growth_function, fertility_brass_polynomial, aging_args, learning_args, growth_args, fertility_args)
fitness_base_b = np.array(fitness_base_b)
t1 = perf_counter()
print(f"Time for grid : {t1-t0}")

np.save('./output/base_b_fitness.npy', fitness_base_b)

t0 = perf_counter()
fitness_reduced_b = parallel_compute_fitness_grid(reduced_b_grid, aging_gompertz_makeham, learning_function, growth_function, fertility_brass_polynomial, aging_args, learning_args, growth_args, fertility_args)
fitness_reduced_b = np.array(fitness_reduced_b)
t1 = perf_counter()
print(f"Time for grid : {t1-t0}")

np.save('./output/reduced_b_fitness.npy', fitness_reduced_b)

fitness_difference = fitness_reduced_b - fitness_base_b

output = pd.DataFrame(np.concatenate((grid, fitness_difference.reshape(-1, 1)), axis = 1), columns = columns)
output.to_csv("./output/fitness_difference.csv", index = False)



