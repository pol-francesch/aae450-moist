# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 11:14:52 2019

@author: NKrvavica
"""

import timeit
import numpy as np
import fqs


def eig_cubic_roots(p):
    '''Finds cubic roots via numerical eigenvalue solver
    `npumpy.linalg.eigvals` from a 3x3 companion matrix'''
    a, b, c = (p[:, 1]/p[:, 0], p[:, 2]/p[:, 0], p[:, 3]/p[:, 0],)
    N = len(a)
    A = np.zeros((N, 3, 3))
    A[:, 1:, :2] = np.eye(2)
    A[:, :, 2] = - np.array([c, b, a]).T
    roots = np.linalg.eigvals(A)
    return roots



# --------------------------------------------------------------------------- #
# Test speed of fqs cubic solver compared to np.roots and np.linalg.eigvals
# --------------------------------------------------------------------------- #

# Number of samples (sets of randomly generated cubic coefficients)
N = 100

# Generate polynomial coefficients
range_coeff = 100
p = np.random.rand(N, 4)*(range_coeff) - range_coeff/2

# number of runs
runs = 10

times = []
for i in range(runs):
    start = timeit.default_timer()
    roots1 = [np.roots(pi) for pi in p]
    stop = timeit.default_timer()
    time = stop - start
    times.append(time)
print('np.roots: {:.4f} ms (best of {} runs)'
      .format(np.array(times).mean()*1_000, runs))

times = []
for i in range(runs):
    start = timeit.default_timer()
    roots2 = eig_cubic_roots(p)
    stop = timeit.default_timer()
    time = stop - start
    times.append(time)
print('np.linalg.eigvals: {:.4f} ms (average of {} runs)'
      .format(np.array(times).mean()*1_000, runs))
print('max err: {:.2e}'.format(abs(np.sort(roots2, axis=1)
                    - (np.sort(roots1, axis=1))).max()))

times = []
for i in range(runs):
    start = timeit.default_timer()
    roots3 = [fqs.single_cubic(*pi) for pi in p]
    stop = timeit.default_timer()
    time = stop - start
    times.append(time)
print('fqs.single_cubic: {:.4f} ms (average of {} runs)'
      .format(np.array(times).mean()*1_000, runs))
print('max err: {:.2e}'.format(abs(np.sort(roots3, axis=1)
                    - (np.sort(roots1, axis=1))).max()))

times = []
for i in range(runs):
    start = timeit.default_timer()
    roots = fqs.multi_cubic(*p.T)
    roots4 = np.array(roots).T
    stop = timeit.default_timer()
    time = stop - start
    times.append(time)
print('fqs.multi_cubic: {:.4f} ms (average of {} runs)'
      .format(np.array(times).mean()*1_000, runs))
print('max err: {:.2e}'.format(abs(np.sort(roots4, axis=1)
                    - (np.sort(roots1, axis=1))).max()))

times = []
for i in range(runs):
    start = timeit.default_timer()
    roots5 = fqs.cubic_roots(p)
    stop = timeit.default_timer()
    time = stop - start
    times.append(time)
print('fqs.cubic_roots: {:.4f} ms (average of {} runs)'
      .format(np.array(times).mean()*1_000, runs))
print('max err: {:.2e}'.format(abs(np.sort(roots5, axis=1)
                    - (np.sort(roots1, axis=1))).max()))
# --------------------------------------------------------------------------- #
