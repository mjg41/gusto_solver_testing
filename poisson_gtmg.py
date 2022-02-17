from firedrake import *
import argparse

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

    m = UnitSquareMesh(10, 10)
    nlevels = 2
    mh = MeshHierarchy(m, nlevels)
    mesh = mh[-1]
    x = SpatialCoordinate(mesh)

    def get_p1_space():
        return FunctionSpace(mesh, "CG", 1)

    def get_p1_prb_bcs():
        return DirichletBC(get_p1_space(), Constant(0.0), "on_boundary")

    def p1_callback():
        P1 = get_p1_space()
        p = TrialFunction(P1)
        q = TestFunction(P1)
        return inner(grad(p), grad(q))*dx

    degree = 1
    RT = FunctionSpace(mesh, "RT", degree)
    DG = FunctionSpace(mesh, "DG", degree - 1)
    W = RT * DG

    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    f = Function(DG)
    f.interpolate(-2*(x[0]-1)*x[0] - 2*(x[1]-1)*x[1])

    a = (inner(sigma, tau) - inner(u, div(tau)) + inner(div(sigma), v))*dx
    L = inner(f, v)*dx

    w = Function(W)

    if (args.coarse_solver == 'exact'):
        coarse_params = {'ksp_type': 'preonly',
                         #'ksp_monitor':None,
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

    solve(a == L, w, solver_parameters=params, appctx=appctx)
    _, uh = w.split()

    # Analytical solution
    f.interpolate(x[0]*(1-x[0])*x[1]*(1-x[1]))

    return errornorm(f, uh, norm_type="L2")

if __name__ == '__main__':
    ''' === M A I N ==== '''
    # Parse command line options
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--coarse_solver',
                        choices=(None,'exact','amg','gmg'),
                        default='gmg',
                        help='select coarse solver to use on nonnested coarse system')
    args, _ = parser.parse_known_args()
    # solve mixed Poisson problem
    run_gtmg_mixed_poisson(args)
