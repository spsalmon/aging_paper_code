import numpy as np
import cython

cdef extern from "math.h":
	double exp(double)

cpdef double aging_gompertz(double x, double a, double b):
	return a * exp(b * x)

cpdef double[:] aging_gompertz_vectorized(double[:] x, double a, double b):
	cdef Py_ssize_t len_x = x.shape[0]
	result = np.zeros((len_x), dtype=np.double)
	cdef double[:] result_view = result
	cdef Py_ssize_t i

	for i in range(len_x):
		result_view[i] = aging_gompertz(x[i], a, b)

	return result

cpdef double aging_gompertz_makeham(double x, double a, double b, double c):
	return c + a * exp(b * x)

cpdef double[:] aging_gompertz_makeham_vectorized(double[:] x, double a, double b, double c):
	cdef Py_ssize_t len_x = x.shape[0]
	result = np.zeros((len_x), dtype=np.double)
	cdef double[:] result_view = result
	cdef Py_ssize_t i

	for i in range(len_x):
		result_view[i] = aging_gompertz_makeham(x[i], a, b, c)

	return result

@cython.cdivision(True)
cpdef double learning_function(double x, double lmax, double k_learning, double n):
	return (lmax/(1 + exp(n*(x-k_learning)))) - lmax

@cython.cdivision(True)
cpdef double[:] learning_function_vectorized(double[:] x, double lmax, double k_learning, double n):
	cdef Py_ssize_t len_x = x.shape[0]
	result = np.zeros((len_x), dtype=np.double)
	cdef double[:] result_view = result
	cdef Py_ssize_t i

	for i in range(len_x):
		result_view[i] = learning_function(x[i], lmax, k_learning, n)

	return result

@cython.cdivision(True)
cpdef double growth_function(double x, double gmax, double growth_rate):
	return (gmax/(1 + (x**growth_rate))) - gmax

@cython.cdivision(True)
cpdef double[:] growth_function_vectorized(double[:] x, double gmax, double growth_rate):
	cdef Py_ssize_t len_x = x.shape[0]
	result = np.zeros((len_x), dtype=np.double)
	cdef double[:] result_view = result
	cdef Py_ssize_t i

	for i in range(len_x):
		result_view[i] = growth_function(x[i], gmax, growth_rate)

	return result

cpdef double fertility_brass_polynomial(double x, double c, double d, double w):
	if (x > d and x < (d+w)):
		return c*(x-d)*((d+w-x)**2)
	return 0.0

cpdef double[:] fertility_brass_polynomial_vectorized(double[:] x, double c, double d, double w):
	cdef Py_ssize_t len_x = x.shape[0]
	result = np.zeros((len_x), dtype=np.double)
	cdef double[:] result_view = result
	cdef Py_ssize_t i

	for i in range(len_x):
		result_view[i] = fertility_brass_polynomial(x[i], c, d, w)

	return result

@cython.cdivision(True)
cpdef double gla_model(double x, aging_function, learning_function, growth_function, aging_args, learning_args, growth_args, double minimum_mortality = 1e-5):
	cdef double aging_result, learning_result, growth_result, gla_result
	aging_result = aging_function(x, *aging_args)
	learning_result = learning_function(x, *learning_args)
	growth_result = growth_function(x, *growth_args)

	gla_result = aging_result + learning_result + growth_result

	if gla_result < 0:
		gla_result = minimum_mortality
	
	return gla_result

@cython.cdivision(True)
cpdef double[:] gla_model_vectorized(double[:] x, aging_function, learning_function, growth_function, aging_args, learning_args, growth_args, double minimum_mortality = 1e-5):
	cdef Py_ssize_t len_x = x.shape[0]
	result = np.zeros((len_x), dtype=np.double)
	cdef double[:] result_view = result
	cdef Py_ssize_t i

	for i in range(len_x):
		result_view[i] = gla_model(x[i], aging_function, learning_function, growth_function, aging_args, learning_args, growth_args, minimum_mortality)

	return result

@cython.cdivision(True)
cpdef double[:] classic_gla_model_vectorized(double[:] x, double a, double b, double c, double lmax, double k_learning, double n, double gmax, double growth_rate):
	cdef Py_ssize_t len_x = x.shape[0]
	result = np.zeros((len_x), dtype=np.double)
	cdef double[:] result_view = result
	cdef Py_ssize_t i

	for i in range(len_x):
		result_view[i] = gla_model(x[i], aging_gompertz_makeham, learning_function, growth_function, (a, b, c), (lmax, k_learning, n), (gmax, growth_rate), 1e-10)
	return result

@cython.cdivision(True)
cpdef double[:] heligman_model_vectorized(double[:] x, double A, double B, double C, double D, double E, double F, double G, double H, eps=1e-10):
	cdef Py_ssize_t len_x = x.shape[0]
	result = np.zeros((len_x), dtype=np.double)
	cdef double[:] result_view = result
	cdef Py_ssize_t i

	for i in range(len_x):
		qx_px_ratio = A**((x[i]+B)**C) + D *np.exp(-E*((np.log(x[i]+eps) - np.log(F))**2)) + G*(H**x[i])
		qx = qx_px_ratio/(1+qx_px_ratio)
		result_view[i] = qx
	return result

@cython.cdivision(True)
cpdef double[:] siler_model_vectorized(double[:] x, double a1, double a2, double a3, double b1, double b3):
	cdef Py_ssize_t len_x = x.shape[0]
	result = np.zeros((len_x), dtype=np.double)
	cdef double[:] result_view = result
	cdef Py_ssize_t i

	for i in range(len_x):
		result_view[i] = a1*np.exp(-b1*x[i]) + a2 + a3*np.exp(b3*x[i])
	return result