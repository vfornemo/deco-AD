import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import jax
from pyscfad.lib import numpy as jnp
from pyscfad import gto
import decodense
from decoAD.hf.dipole import dipole1, dipole2


# Criteria of decomposition
e_decomp1 = decodense.DecompCls(part='atoms', mo_basis='pm', prop='energy', verbose=0, pop_method='iao')
dip_decomp1 = decodense.DecompCls(part='atoms', mo_basis='pm', prop='dipole', verbose=0, pop_method='iao')
e_decomp2 = decodense.DecompCls(part='atoms', mo_basis='can', prop='energy', verbose=0, pop_method='iao')
dip_decomp2 = decodense.DecompCls(part='atoms', mo_basis='can', prop='dipole', verbose=0, pop_method='iao')
# Static external electric field
E0 = jnp.array([0., 0., 0.])

# NH3 Molecule
mol = gto.Mole()
mol.atom = '''
N
H 1 1.008000
H 1 1.008000 2 109.47
H 1 1.008000 2 109.47 3 120
'''

# N
# H 1 1.008000
# H 1 1.008000 2 109.47
# H 1 1.008000 2 109.47 3 120

mol.basis = 'aug-pcseg-1'
mol.build(trace_exp=False, trace_ctr_coeff=False)

print('\n###### NH3 ######\n')

dip1_pm = dipole1(E0,e_decomp1,mol)
dip1_can = dipole1(E0,e_decomp2,mol)
print('JAX PM/IAO Dipole',jnp.sum(dip1_pm,0))
print('JAX can/IAO Dipole',jnp.sum(dip1_can,0))

dip2_pm = dipole2(E0,dip_decomp1,mol)
dip2_can = dipole2(E0,dip_decomp2,mol)
print('Decodense PM/IAO Dipole',jnp.sum(dip2_pm,0))
print('Decodense can/IAO Dipole',jnp.sum(dip2_can,0))

pol1_pm = jax.jacrev(dipole1)(E0,e_decomp1,mol)
pol2_pm = jax.jacrev(dipole2)(E0,dip_decomp1,mol)

pol1_can = jax.jacrev(dipole1)(E0,e_decomp2,mol)
pol2_can = jax.jacrev(dipole2)(E0,dip_decomp2,mol)

# print('1st derivative PM/IAO Polarizability',jnp.trace(jnp.sum(pol2_can,0))/3)
# print('2nd derivative PM/IAO Polarizability',jnp.trace(jnp.sum(pol1_can,0))/3)
# print('1st derivative PM/IAO Polarizability',jnp.trace(pol2_pm)/3)
# print('2nd derivative PM/IAO Polarizability',jnp.trace(pol1_pm)/3)
# print('1st derivative can/IAO Polarizability',jnp.trace(pol2_can)/3)
# print('2nd derivative can/IAO Polarizability',jnp.trace(pol1_can)/3)

trace1 = [jnp.trace(x) for x in pol1_pm/3]
trace2 = [jnp.trace(x) for x in pol2_pm/3]
trace3 = [jnp.trace(x) for x in pol1_can/3]
trace4 = [jnp.trace(x) for x in pol2_can/3]

print('1st derivative PM/IAO Polarizability',(pol2_pm)/3, trace2)
print('2nd derivative PM/IAO Polarizability',(pol1_pm)/3, trace1)
print('1st derivative can/IAO Polarizability',(pol2_can)/3, trace4)
print('2nd derivative can/IAO Polarizability',(pol1_can)/3, trace3)



