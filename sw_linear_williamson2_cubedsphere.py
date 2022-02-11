from gusto import *
# from firedrake import (CubedSphereMesh, SpatialCoordinate,
#                        as_vector, FunctionSpace, TrialFunction,
#                        TestFunction, inner, grad, dx, dS)
from firedrake import *
from math import pi
import numpy as np
from mpi4py import MPI
import sys
import argparse
import logging
from ksp_monitor import *
from transfer_kernels import prolongation_matrix

from firedrake.petsc import PETSc
PETSc.Sys.popErrorHandler()

def pprint(*s):
    '''Print string only on master process

    :arg s: Stuff to print
    '''
    if (MPI.COMM_WORLD.Get_rank() == 0):
        print (*s,flush=True)

comm_size = MPI.COMM_WORLD.Get_size()
pprint ('Running on '+str(comm_size)+' processors')

# Parse command line options
parser = argparse.ArgumentParser(allow_abbrev=False)

parser.add_argument('--testing',
                    action='store_true',
                    default=False,
                    help='testing runs only for a short time without convergence test')

parser.add_argument('--degree',
                    type=int,
                    default=1,
                    help='polynomial degree of DG space')

parser.add_argument('--mesh_degree',
                    type=int,
                    default=3,
                    help='polynomial degree of mesh')

parser.add_argument('--tfinal',
                    type=float,
                    default=1.0,
                    help='final time (in days)')

parser.add_argument('--nrefine',
                    type=int,
                    default=3,
                    help='mesh refinement level, has to be > 1 for flat geometry')

parser.add_argument('--solver',
                    choices=('direct','pressure_multigrid','hybridised_amg','hybridised_nonnested'),
                    default='direct',
                    help='select solver to use for solving linear system')

parser.add_argument('--solver_rtol',
                    type=float,
                    default=1.0E-8,
                    help='Relative tolerance to use in solver')

parser.add_argument('--coarse_solver',
                    choices=(None,'exact','amg','gmg'),
                    default=None,
                    help='select coarse solver to use on nonnested coarse system')

parser.add_argument('--coarse_space',
                    choices=(None,'DG0','P1'),
                    default=None,
                    help='select coarse space to use on nonnested coarse system')

parser.add_argument('--ksp_verbosity',
                    type=int,
                    default=0,
                    help='verbosity of KSP solvers. 0: no output, 1: summary, 2: full output')

args, unknown = parser.parse_known_args()

# Raise error if coarse solver and coarse space haven't been specified
# for non-nested hybridised solver
if (args.solver == 'hybridised_nonnested'):
    if not args.coarse_solver:
        raise Exception('Need to specify coarse solver for: '+args.solver)
    if not args.coarse_space:
        raise Exception('Need to specify coarse space for: '+args.solver)

# Day in seconds
day = 24.*60.*60.

# setup resolution and timestepping parameters for convergence test
if args.testing:
    ref_dt = {args.nrefine: 3000.}
    tmax = 3000.
else:
    # setup resolution and timestepping parameters for convergence test
    ref_dt = {(args.nrefine): 4000., (args.nrefine + 1): 2000.,
            (args.nrefine + 2): 1000., (args.nrefine + 3): 500.}
    tmax = args.tfinal*day


# setup shallow water parameters
R = 6371220.
H = 5960.

# setup input that doesn't change with ref level or dt
fieldlist = ['u', 'D']
parameters = ShallowWaterParameters(H=H)
cg = day / R * np.sqrt(parameters.g*H)

# setup solver parameters

# Set up default appctx
appctx = {}

# Set up solver
if (args.solver == 'direct'):
    solver_parameters = {'mat_type': 'aij',
                         'ksp_type': 'preonly',
                        #  'ksp_view': None,
                         'pc_type': 'lu',
                         'pc_factor_mat_solver_type': 'mumps'}
elif (args.solver == 'pressure_multigrid'):
    solver_parameters = {'ksp_type': 'gmres',
                    'ksp_rtol': args.solver_rtol,
                    'pc_type': 'fieldsplit',
                    'pc_fieldsplit_type': 'schur',
                    'pc_fieldsplit_schur_fact_type': 'FULL',
                    'pc_fieldsplit_schur_precondition': 'selfp',
                    'fieldsplit_0': {'ksp_type': 'preonly',
                                    'pc_type': 'bjacobi',
                                    'sub_pc_type': 'ilu'},
                    'fieldsplit_1': {'ksp_type': 'preonly',
                                    'pc_type': 'hypre',
                                    'pc_mg_log': None,
                                    'mg_levels': {'ksp_type': 'richardson',
                                                    'ksp_richardson_scale':0.6,
                                                    'ksp_max_it': 2,
                                                    'pc_type':'jacobi'}}} 
elif (args.solver == 'hybridised_amg'):
    solver_parameters = {'ksp_type': 'preonly',
                    'mat_type': 'matfree',
                    'pc_type': 'python',
                    'pc_python_type': 'firedrake.HybridizationPC',
                    # Solver for the trace system
                    'hybridization': {'ksp_type': 'bcgs',
                                        'pc_type': 'hypre',
                                        'ksp_rtol': args.solver_rtol,
                                        'mg_levels': {'ksp_type': 'richardson',
                                                    'ksp_richardson_scale':0.6,
                                                    'ksp_max_it': 2,
                                                    'pc_type': 'jacobi'}}}
elif (args.solver == 'hybridised_nonnested'):
    if (args.coarse_solver == 'exact'):
        coarse_param = {'ksp_type': 'preonly',
                        'pc_type': 'lu'}
    elif (args.coarse_solver == 'amg'):
        coarse_param = {'ksp_type': 'preonly',
                        'pc_type': 'hypre',
                        'pc_gamg_sym_graph': True,
                        'mg_levels': {'ksp_type': 'richardson',
                                                  'ksp_max_it': 2,
                                                  'pc_type': 'bjacobi',
                                                  'sub_pc_type': 'ilu'}}
    elif (args.coarse_solver == 'gmg'):
        coarse_param = {'ksp_type': 'preonly',
                        'pc_type': 'mg',
                        'pc_mg_cycle_type': 'v',
                        'mg_levels': {'ksp_type': 'chebyshev',
                                      'ksp_max_it': 2,
                                      'pc_type': 'bjacobi',
                                      'sub_pc_type': 'sor'},
                        # 'mg_coarse': {'ksp_type': 'chebyshev',
                        #               'ksp_max_it': 2,
                        #               'pc_type': 'bjacobi',
                        #               'sub_pc_type': 'sor'}}
                        }
    else:
        raise Exception('Unknown coarse solver type: ' + args.coarse_solver)            
    
    solver_parameters = {'ksp_type': 'preonly',
                         'mat_type': 'matfree',
                         'pc_type': 'python',
                         'pc_python_type': 'firedrake.HybridizationPC',
                         # Solver for the trace system
                         'hybridization': {'ksp_type': 'bcgs',
                                           'pc_type': 'python',
                                           'ksp_rtol': args.solver_rtol,
                                           'pc_python_type': 'firedrake.GTMGPC',
                                           'gt': {'mat_type': 'aij',
                                                  'pc_mg_log': None,
                                                  'mg_levels': {'ksp_type': 'chebyshev',
                                                                #'ksp_richardson_scale':0.6,
                                                                'ksp_max_it': 2,
                                                                'pc_type': 'sor'},
                                                  'mg_coarse': coarse_param}}}
else:
    raise Exception('Unknown solver type: '+args.solver)

for ref_level, dt in ref_dt.items():

    # Setup output directory
    dirname = "sw_W2_ref%s_dt%s" % (ref_level, dt)

    # Define mesh - need a mesh hierarchy for gmg
    m = CubedSphereMesh(radius=R,
                        refinement_level=ref_level,
                        degree=args.mesh_degree)
    nlevels = 1
    mh = MeshHierarchy(m, nlevels)
    mesh = mh[-1]
    x = SpatialCoordinate(mesh)
    global_normal = x
    mesh.init_cell_orientations(x)

    #Set timestep
    timestepping = TimesteppingParameters(dt=dt)

    # Compute mesh properties
    ncell = 6*4**ref_level
    h = np.sqrt(4.*np.pi/ncell)

    # Print out parameters of run
    pprint (" nrefine           = ", ref_level)
    pprint (" degree p          = ", args.degree)
    pprint (" mesh degree p     = ", args.mesh_degree)
    pprint (" c_g               =  {:.3e}".format(cg))
    pprint (" f/c_g             =  {:.3e}".format(2.*parameters.Omega/cg))
    pprint (" h                 =  {:.3e}".format(h))
    pprint (" dt                = ", dt)
    pprint (" T                 = ", args.tfinal)
    pprint (" #cells            = ", ncell)
    pprint (" solver            = ", args.solver)

    if (args.solver == 'hybridised_nonnested'):
        pprint(" coarse solver      = ", args.coarse_solver)
        pprint(" coarse space       = ", args.coarse_space)
    
    # Print out number of processors
    pprint(" Number of processes = ",mesh.comm.size)

    output = OutputParameters(dirname=dirname,
                              dumplist_latlon=['D', 'D_error'],
                              steady_state_error_fields=['D', 'u'],
                              log_level='INFO')

    diagnostic_fields = [RelativeVorticity(), PotentialVorticity(),
                         ShallowWaterKineticEnergy(),
                         ShallowWaterPotentialEnergy(),
                         ShallowWaterPotentialEnstrophy()]

    state = State(mesh, horizontal_degree=args.degree,
                  family="RTCF",
                  timestepping=timestepping,
                  output=output,
                  parameters=parameters,
                  diagnostic_fields=diagnostic_fields,
                  fieldlist=fieldlist)

    # Now that we have the mesh and the state we can compute the
    # coarse space and the coarse callback if we're using the 
    # hybridised non-nested solver

    if (args.solver == 'hybridised_nonnested'):

        if (args.coarse_space == 'P1'):

            # Define P1 coarse space and callback

            def get_coarse_space():
                return FunctionSpace(mesh, 'CG', 1)

            def coarse_callback():
                P1 = get_coarse_space()
                q = TrialFunction(P1)
                r = TestFunction(P1)
                beta = dt*state.timestepping.alpha

                return (inner(q, r) +
                        beta**2*
                        parameters.g*
                        parameters.H*
                        inner(grad(q), grad(r)))*dx
    
        elif (args.coarse_space == 'DG0'):

            # Define DG0 coarse space and callback

            def get_coarse_space():
                return FunctionSpace(mesh, 'DG', 0)

            def coarse_callback():
                DG0 = get_coarse_space()
                phi = TrialFunction(DG0)
                psi = TestFunction(DG0)
                h = 0.5**args.nrefine
                beta = dt*state.timestepping.alpha

                return psi*phi*dx + (beta/h)*(psi('+')-psi('-'))*(phi('+')-phi('-'))*dS
        else:
            raise Exception('Unknown coarse space: '+args.coarse_space)

        V_trace = FunctionSpace(mesh, "HDiv Trace", args.degree)
        interpolation_matrix = prolongation_matrix(V_trace,get_coarse_space())

        appctx = {'get_coarse_operator': coarse_callback,
                  'get_coarse_space': get_coarse_space,
                  'interpolation_matrix':interpolation_matrix}

    # interpolate initial conditions
    u0 = state.fields("u")
    D0 = state.fields("D")
    x = SpatialCoordinate(mesh)
    u_max = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)
    uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
    Omega = parameters.Omega
    g = parameters.g
    Dexpr = - ((R * Omega * u_max)*(x[2]*x[2]/(R*R)))/g
    # Coriolis expression
    fexpr = 2*Omega*x[2]/R
    V = FunctionSpace(mesh, "CG", 1)
    f = state.fields("coriolis", V)
    f.interpolate(fexpr)  # Coriolis frequency (1/s)

    u0.project(uexpr)
    D0.interpolate(Dexpr)
    state.initialise([('u', u0),
                      ('D', D0)])

    Deqn = LinearAdvection(state, D0.function_space(), state.parameters.H, ibp=IntegrateByParts.ONCE, equation_form="continuity")
    advected_fields = []
    advected_fields.append(("u", NoAdvection(state, u0, None)))
    advected_fields.append(("D", ForwardEuler(state, D0, Deqn)))

    # Do this before if hybridized utilising custom monitor hook in
    # HybridizationPC
    # Create custom monitor and add to appctx

    if 'hybridised' in args.solver:
        ksp_monitor = KSPMonitor(label='hybridised_linear_solve',
                                 comm=mesh.comm,
                                 verbose=args.ksp_verbosity)
        appctx['custom_monitor'] = ksp_monitor


    linear_solver = ShallowWaterSolver(state,
                                       solver_parameters=solver_parameters,
                                       overwrite_solver_parameters=True,
                                       appctx=appctx)

    # If hybridised also monitor trace convergence

    if 'hybridised' not in args.solver:
        ksp_monitor = KSPMonitor(label='linear_solve',
                                 comm=mesh.comm,
                                 verbose=args.ksp_verbosity)
        linear_solver.uD_solver.snes.ksp.setMonitor(ksp_monitor)

    # Set up forcing
    sw_forcing = ShallowWaterForcing(state, linear=True)

    # build time stepper
    stepper = CrankNicolson(state, advected_fields, linear_solver,
                            sw_forcing)

    with ksp_monitor:

        stepper.run(t=0, tmax=tmax)
