import numpy as np
import scipy.integrate
import scipy.optimize
from .gla import gla_model

cdef extern from "math.h":
	double exp(double)

def survival_l(double x, aging_function, learning_function, growth_function, list aging_args, list learning_args, list growth_args):
	cdef double cumulative_hazard = scipy.integrate.quad(lambda t: gla_model(t, aging_function, learning_function, growth_function, aging_args, learning_args, growth_args), 0, x, limit=100)[0]
	return exp(-cumulative_hazard)

def euler_lotka_function(double r, aging_function, learning_function, growth_function, fertility_function, list aging_args, list learning_args, list growth_args, list fertility_args):
	cdef double integral = scipy.integrate.quad(lambda t: fertility_function(t, *fertility_args) * survival_l(t, aging_function, learning_function, growth_function, aging_args, learning_args, growth_args) * np.exp(-r*t), 0, 100, limit=100)[0]
	return integral - 1

def compute_fitness(aging_function, learning_function, growth_function, fertility_function, list aging_args, list learning_args, list growth_args, list fertility_args):
	cdef double fitness
	fitness = scipy.optimize.fsolve(lambda r: euler_lotka_function(r, aging_function, learning_function, growth_function, fertility_function, aging_args, learning_args, growth_args, fertility_args), 0.01)[0]
	if fitness < 0:
		return 0.0
	return fitness

cpdef double compute_fitness_grid_point(grid_point, aging_function, learning_function, growth_function, fertility_function, list aging_args, list learning_args, list growth_args, list fertility_args):
	cdef double fitness
	new_aging_args = [aging_args[0], grid_point[0], aging_args[2]]
	new_learning_args = [grid_point[1], learning_args[1], learning_args[2]]
	fitness = compute_fitness(aging_function, learning_function, growth_function, fertility_function, new_aging_args, new_learning_args, growth_args, fertility_args)
	return fitness

cpdef double[:] compute_fitness_grid(double[:, :] grid, aging_function, learning_function, growth_function, fertility_function, list aging_args, list learning_args, list growth_args, list fertility_args):
	
	cdef Py_ssize_t x_max = grid.shape[0]
	fitness = np.zeros((x_max), dtype=np.double)
	cdef double[:] fitness_view = fitness

	cdef Py_ssize_t i

	for i in range(x_max):
		fitness_view[i] = compute_fitness_grid_point(grid[i], aging_function, learning_function, growth_function, fertility_function, aging_args, learning_args, growth_args, fertility_args)

	return fitness