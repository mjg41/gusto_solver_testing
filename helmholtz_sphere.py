from firedrake import *
import argparse
import transfer_kernels
from ksp_monitor import *
import numpy as np


from firedrake.petsc import PETSc
PETSc.Sys.popErrorHandler()

'''
=======================================================
GTMG solve of the helmholtz equation
=======================================================

adapted from test_mixed_poisson_gtmg.py in the
firedrake/tests/multigrid directory. This MWE is used to debug issues
with the GTMG preconditioner in the gusto code.
'''

def run_gtmg_helmholtz(args):
    '''Solve helmholtz equation on the sphere

    :arg args: command line arguments controlling solver options
    '''
    R = 6371220.
    mesh = CubedSphereMesh(radius=R, refinement_level=args.nrefine, degree=3)

    # Compute mesh properties
    ncell = 6*4**args.nrefine
    h = np.sqrt(4.*np.pi/ncell)

    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)
    V = FunctionSpace(mesh, 'RTCF', args.degree+1)
    Q = FunctionSpace(mesh, 'DG', args.degree)
    W = V*Q

    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    f = Function(Q)
    f.interpolate(cos((x[0]/R) + sin(x[0]/R) + cos(x[0]/R)*sin(x[0]/R)))
    a = (inner(p, q) - h*inner(div(u), q) + inner(u, v) + h*inner(p, div(v))) * dx
    L = inner(f, q) * dx

    w = Function(W)

    def get_p1_space():
        return FunctionSpace(mesh, "CG", 1)

    def p1_callback():
        P1 = get_p1_space()
        p = TrialFunction(P1)
        q = TestFunction(P1)
        return 1E-6*(p*q + h**2*inner(grad(p), grad(q)))*dx

    if (args.coarse_solver == 'exact'):
        coarse_params = {'ksp_type': 'preonly',
                         'ksp_monitor':None,
                         'pc_type': 'lu'}
    elif (args.coarse_solver == 'amg'):
        coarse_params = {'ksp_type': 'preonly',
                         'pc_type': 'hypre',
                         'pc_gamg_sym_graph': True,
                         'mg_levels': {'ksp_type': 'richardson',
                                                   'ksp_max_it': 2,
                                                   'pc_type': 'bjacobi',
                                                   'sub_pc_type': 'ilu'}}

    elif (args.coarse_solver == 'gmg'):
        coarse_params = {'ksp_type': 'preonly',
                         'pc_type': 'mg',
                         'pc_mg_type': 'full',
                         'mg_levels': {'ksp_type': 'chebyshev',
                                       'pc_type': 'jacobi',
                                       'ksp_max_it': 3}}
    else:
        raise Exception('Unknown coarse solver type: ' + args.coarse_solver)

    params = {'mat_type': 'matfree',
              'ksp_type': 'preonly',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.HybridizationPC',
              'hybridization': {'ksp_type': 'cg',
                                'mat_type': 'matfree',
                                'ksp_monitor':None,
                                'pc_type': 'python',
                                'pc_python_type': 'firedrake.GTMGPC',
                                'gt': {'mg_levels': {'ksp_type': 'chebyshev',
                                                     'pc_type': 'jacobi',
                                                     'ksp_max_it': 3},
                                       'mg_coarse': coarse_params}}}

    appctx = {'get_coarse_operator': p1_callback,
              'get_coarse_space': get_p1_space}
    if args.custom_transfer:
        V_trace = FunctionSpace(mesh, "HDiv Trace", args.degree)
        interp_matrix = transfer_kernels.prolongation_matrix(V_trace,
                                                             get_p1_space())
        appctx['interpolation_matrix'] = interp_matrix

    ksp_monitor = KSPMonitor(label='hybridised_linear_solve',
                             comm=mesh.comm,
                             verbose=2)
    appctx['custom_monitor'] = ksp_monitor

    # Provide a callback to construct the trace nullspace
    # def nullspace_basis(T):
    #     return VectorSpaceBasis(constant=True)

    # appctx['trace_nullspace'] =  nullspace_basis

    with ksp_monitor:

        solve(a == L, w, solver_parameters=params, appctx=appctx)

    _, uh = w.split()

    # Analytical solution
    # f.interpolate(x[0]*(1-x[0])*x[1]*(1-x[1]))

    return errornorm(f, uh, norm_type="L2")

if __name__ == '__main__':
    ''' === M A I N ==== '''
    # Parse command line options
    parser = argparse.ArgumentParser(allow_abbrev=False)
    # coarse level solver
    parser.add_argument('--coarse_solver',
                        choices=('exact','amg','gmg'),
                        default='gmg',
                        help='select coarse solver to use on nonnested coarse system')
    # use custom transfer operators from transfer_kernels.py?
    parser.add_argument('--custom_transfer',
                        action='store_true',
                        default=False,
                        help='use custom transfer operators from transfer_kernels.py?')
    # Polynomial degree of velocity space
    parser.add_argument('--degree',
                        type=int,
                        default=1,
                        help='polynomial degree of velocity space')
    # Mesh to use
    parser.add_argument('--nrefine',
                        type=int,
                        default=3,
                        help='mesh refinement level')

    args, _ = parser.parse_known_args()
    print ('===== parameters =====')
    print ('coarse solver    = ',args.coarse_solver)
    print ('degree           = ',args.degree)
    print ('mesh refinement  = ',args.nrefine)
    print ('custom transfer? = ',args.custom_transfer)
    print ('')
    # solve mixed Poisson problem
    run_gtmg_helmholtz(args)
