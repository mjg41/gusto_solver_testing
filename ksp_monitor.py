import numpy as np
import time

class KSPMonitor(object):
    '''KSP Monitor writing to stdout.

    :arg label: Name of solver
    :arg verbose: verbosity. 0=print nothing, 1=only print summary of results, 2=print everything in detail
    '''
    def __init__(self,label='',verbose=2,comm=None):
        self.label = label
        self.verbose = verbose
        self.initial_residual = 1.0
        self.iterations = []
        self.resnorm = []
        self.t_start = 0.0
        self.t_start_iter = 0.0
        self.t_finish = 0.0
        self.its = 0
        self.niter= []
        self._comm = comm
        self.list_of_norms = []

    '''Call logger. 

    This method is called by the KSP class and should write the output.
    
    :arg ksp: The calling ksp instance
    :arg its: The current iterations
    :arg rnorm: The current residual norm
    '''
    def __call__(self,ksp,its,rnorm):
        if (its==0):
            self.rnorm0 = rnorm
            self.its-=1
            if (self.verbose>=2):
                print()
        if (self.verbose>=2):
            s = '  KSP '+('%20s' % self.label)
            s += ('  %6d' % its)+' : '
            s += ('  %10.6e' % rnorm)
            s += ('  %10.6e' % (rnorm/self.rnorm0))
            if (its > 0):
                s += ('  %8.4f' % (rnorm/self.rnorm_old))
            else:
                s += '      ----'
            self._print(s)
        self.its += 1
        if (self.its == 1):
            self.t_start_iter = time.process_time()
        self.rnorm = rnorm
        self.iterations.append(its)
        self.resnorm.append(rnorm)
        self.rnorm_old = rnorm

    def __enter__(self):
        '''Print information at beginning of iteration.
        '''
        self.iterations = []
        self.resnorm = []
        self.its = 0
        if (self.verbose >= 1):
            self._print('')
        if (self.verbose == 2):
            s = '  KSP '+('%20s' % self.label)
            s += '    iter             rnrm   rnrm/rnrm_0       rho'
            self._print(s)
        self.t_start = time.process_time()
        self.rnorm = 1.0
        self.rnorm0 = 1.0
        self.rnorm_old = 1.0
        return self
    
    def __exit__(self,*exc):
        '''Print information at end of iteration.
        '''
        self.t_finish = time.process_time()
        niter = self.its
        if (niter == 0):
            niter = 1
        if (self.verbose == 1):
            s = '  KSP '+('%20s' % self.label)
            s += '    iter             rnrm   rnrm/rnrm_0   rho_avg'
            self._print(s)
            s = '  KSP '+('%20s' % self.label)
            s += ('  %6d' % niter)+' : '
            s += ('  %10.6e' % self.rnorm)
            # Hack to avoid dividing by 0
            if self.rnorm0 == 0:
                self.rnorm0 = 1E-16
            s += ('  %10.6e' % (self.rnorm/self.rnorm0))
            s += ('  %8.4f' % (self.rnorm/self.rnorm0)**(1./float(niter)))
            self._print(s)
        if (self.verbose >= 1):
            t_elapsed = self.t_finish - self.t_start
            t_elapsed_iter = self.t_finish - self.t_start_iter
            s = '  KSP '+('%20s' % self.label)
            s += (' n_iter = %4d' % niter)
            s += (' t_solve = %8.4f s' % t_elapsed)
            s += (' t_iter = %8.4f s' % (t_elapsed_iter/niter))
            s += (' [%8.4f s' % (self.t_start_iter-self.t_start))+']'
            self._print(s)
            self._print('')
        self.niter.append(niter)

    def _print(self,*s):
        '''Print only on master processor
        
        :arg s: stuff to print
        '''
        if (self._comm is None):
            print (*s)
        else:
            if (self._comm.rank == 0):
                print (*s)

    def add_reconstructor(self, foo):
        pass

class KSPMonitorDummy(object):
    def __init__(self):
        pass

    def __call__(self,ksp,its,rnorm):
        pass

    def __enter__(self):
        pass

    def __exit__(self,*exc):
        pass
