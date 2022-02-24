from firedrake import *
import argparse
import transfer_kernels
from ksp_monitor import *


from firedrake.petsc import PETSc
PETSc.Sys.popErrorHandler()

'''
=======================================================
GTMG solve of the mixed Poisson equation
=======================================================

adapted from test_mixed_poisson_gtmg.py in the
firedrake/tests/multigrid directory. This MWE is used to debug issues
with the GTMG preconditioner in the gusto code.
'''

def run_gtmg_mixed_poisson(args):
    '''Solve mixed Poisson equation

    :arg args: command line arguments controlling solver options
    '''
    if (args.mesh == 'unitsquare'):
        m = UnitSquareMesh(10, 10)
        # zero_order_term = False
    elif (args.mesh == 'square'):
        L=10000.
        m = SquareMesh(10, 10, L)
        # zero_order_term = False
        h = m.cell_sizes([L/2,L/2]) # get example delta-x from middle of domain
    elif (args.mesh == 'icosahedralsphere'):
        R = 1.
        m = IcosahedralSphereMesh(radius=R, refinement_level=2, degree=3)
        # zero_order_term = False
    else:
        raise Exception('Unknown mesh: ' + args.mesh)

    nlevels = 2
    mh = MeshHierarchy(m, nlevels)
    mesh = mh[-1]
    x = SpatialCoordinate(mesh)

    # Orient spherical mesh
    if ('sphere' in args.mesh):
        mesh.init_cell_orientations(x)

    def get_p1_space():
        return FunctionSpace(mesh, "CG", 1)

    def get_p1_prb_bcs():
        return DirichletBC(get_p1_space(), Constant(0.0), "on_boundary")

    def p1_callback():
        P1 = get_p1_space()
        p = TrialFunction(P1)
        q = TestFunction(P1)
        return (p*q + h**2*inner(grad(p), grad(q)))*dx

    RT = FunctionSpace(mesh, "RT", args.degree)
    DG = FunctionSpace(mesh, "DG", args.degree - 1)
    W = RT * DG

    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    f = Function(DG)
    f.interpolate(-2*(x[0]-1)*x[0] - 2*(x[1]-1)*x[1])

    a = (inner(sigma, tau) - h*inner(u, div(tau)) + h*inner(div(sigma), v) + inner(u,v))*dx
    L = inner(f, v)*dx

    w = Function(W)

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
              'get_coarse_space': get_p1_space,
              'coarse_space_bcs': get_p1_prb_bcs()}
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
    f.interpolate(x[0]*(1-x[0])*x[1]*(1-x[1]))

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
    parser.add_argument('--mesh',
                        choices=('unitsquare','icosahedralsphere','square'),
                        default='unitsquare',
                        help='mesh to use')

    args, _ = parser.parse_known_args()
    print ('===== parameters =====')
    print ('coarse solver    = ',args.coarse_solver)
    print ('degree           = ',args.degree)
    print ('mesh             = ',args.mesh)
    print ('custom transfer? = ',args.custom_transfer)
    print ('')
    # solve mixed Poisson problem
    run_gtmg_mixed_poisson(args)
