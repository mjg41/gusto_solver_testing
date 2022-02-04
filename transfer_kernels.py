from firedrake.ufl_expr import TestFunction, TrialFunction
from firedrake.function import Function
from firedrake.assemble import assemble
from firedrake.slate import Tensor
from ufl import inner, dS, avg

def prolongation_matrix(Vf,Vc):
    '''Construct PETSc matrix for prolongation

    :arg Vf: fine function space
    :arg Vc: coarse function space
    '''
    u_trial = TrialFunction(Vf)
    u_c_trial = TrialFunction(Vc)
    u_test = TestFunction(Vf)
    a_mass = inner(u_trial('+'),u_test('+'))*dS
    a_proj = inner(avg(u_c_trial),u_test('+'))*dS
    a_proj_mat = assemble(a_proj,mat_type='aij').M.handle
    a_mass_inv = assemble(Tensor(a_mass).inv,mat_type='aij')
    # Rescale by a factor 0.5, since the above produces 2x the mass matrix
    a_mass_inv_mat = a_mass_inv.M.handle
    a_mass_inv_mat.scale(0.5)
    __P_mat = a_mass_inv_mat.matMult(a_proj_mat)
    return __P_mat

def restrict(fine, coarse):
    '''Restrict fine level function

    :arg fine: fine level function to restrict
    :arg coarse: coarse level function
    '''
    Vf = fine.function_space()
    Vc = coarse.function_space()
    fine_tmp = Function(Vf)
    P_mat = prolongation_matrix(Vf,Vc)
    with fine.dat.vec_ro as u:
        with coarse.dat.vec as w:
            P_mat.multTranspose(u,w)
    return coarse


def prolong(coarse, fine):
    '''Prolongate coarse level function

    :arg coarse: coarse level function to prolongate
    :arg fine: fine level function
    '''
    Vf = fine.function_space()
    Vc = coarse.function_space()
    fine_tmp = Function(Vf)
    P_mat = prolongation_matrix(Vf,Vc)
    with coarse.dat.vec_ro as u:
        with fine.dat.vec as w:
            P_mat.mult(u,w)
