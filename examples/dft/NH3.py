import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import jax
from pyscfad.lib import numpy as jnp
from pyscfad import gto
import decodense
from decoAD.dft.dipole import dipole1, dipole2


# Criteria of decomposition
e_decomp1 = decodense.DecompCls(part='atoms', mo_basis='can', prop='energy', verbose=0, pop_method='iao')
dip_decomp1 = decodense.DecompCls(part='atoms', mo_basis='can', prop='dipole', verbose=0, pop_method='iao')
# Static external electric field
E0 = jnp.array([0., 0., 0.])

# NH3 Molecule
mol = gto.Mole()
mol.atom = '''
N  -0.0116    1.0048    0.0076 
H   0.0021   -0.0041    0.0020 
H   0.9253    1.3792    0.0006 
H  -0.5500    1.3634   -0.7668 
'''

# N
# H 1 1.008000
# H 1 1.008000 2 109.47
# H 1 1.008000 2 109.47 3 120

mol.basis = 'sto-3g'
mol.build(trace_exp=False, trace_ctr_coeff=False)

print('\n###### NH3 ######\n')

dip1_can = dipole1(E0,e_decomp1,mol)
print('JAX PM/IAO Dipole',jnp.sum(dip1_can,0))

dip2_can = dipole2(E0,dip_decomp1,mol)
print('Decodense PM/IAO Dipole',jnp.sum(dip2_can,0))

pol1_can = jax.jacrev(dipole1)(E0,e_decomp1,mol)
pol2_can = jax.jacrev(dipole2)(E0,dip_decomp1,mol)
print('1st derivative PM/IAO Polarizability',jnp.trace(jnp.sum(pol2_can,0))/3)
print('2nd derivative PM/IAO Polarizability',jnp.trace(jnp.sum(pol1_can,0))/3)