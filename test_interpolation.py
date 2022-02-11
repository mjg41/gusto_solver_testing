from firedrake import *
from math import pi
import numpy as np
from mpi4py import MPI
import sys
import argparse
import logging
from ksp_monitor import *
from transfer_kernels import prolongation_matrix
import scipy.sparse as sp
import matplotlib.pyplot as plt

from firedrake.petsc import PETSc
PETSc.Sys.popErrorHandler()

# Define mesh
mesh = CubedSphereMesh(radius=1.0,
                       refinement_level=3,
                       degree=2)
x = SpatialCoordinate(mesh)
global_normal = x
mesh.init_cell_orientations(x)

# Create function space
degree = 1
Q = FunctionSpace(mesh, 'CG', degree)

# Create smooth function

def gaussian(x_0, y_0, sigma_x, sigma_y, A):
    return A*exp(-((x[0]-x_0)**2/(2*sigma_x**2) + (x[1]-y_0)**2/(2*sigma_y**2)))

sum_of_gaussians = (gaussian(0., 0., 1., 1., 10.) + 
                   gaussian(0.5, 0.25, 3., 3., 15.) +
                   gaussian(-0.4, 1.3, 0.5, 0.75, 30.) +
                   gaussian(-0.9, -0.85, 5., 3., 5.))

f = Function(Q).project(sum_of_gaussians)

# Now apply interpolation to sum of gaussians

V_trace = FunctionSpace(mesh, "HDiv Trace", degree)
interpolation_matrix = prolongation_matrix(V_trace,Q)

# Construct corresponding scipy sparse matrix

indptr, indices, data = interpolation_matrix.getValuesCSR()
interp_mat_scipy = sp.csr_matrix((data, indices, indptr), shape=interpolation_matrix.getSize())

# Get out function values
f_data = f.dat.data

# Prolong and then apply transpose to restrict
prolong = interp_mat_scipy*f_data
restrict = interp_mat_scipy.T*prolong

# Compare by plotting f_data and prolongated & restricted data

plt.plot(f_data, label='Original function')
plt.plot(restrict, label='After Prolongation & Restriction')
plt.xlabel('dofs')
plt.ylabel('dof values')
plt.legend()
plt.show()
