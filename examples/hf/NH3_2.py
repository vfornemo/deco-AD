import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import jax
from pyscfad.lib import numpy as jnp
from pyscfad import gto
import decodense
from decoAD.hf.dipole import dipole1, dipole2


# Criteria of decomposition
e_decomp1 =   decodense.DecompCls(part='atoms', mo_basis='can', prop='energy', verbose=0, pop_method='iao')
dip_decomp1 = decodense.DecompCls(part='atoms', mo_basis='can', prop='dipole', verbose=0, pop_method='iao')
e_decomp2 =   decodense.DecompCls(part='atoms', mo_basis='fb', prop='energy', verbose=0, pop_method='iao')
dip_decomp2 = decodense.DecompCls(part='atoms', mo_basis='fb', prop='dipole', verbose=0, pop_method='iao')
e_decomp3 =   decodense.DecompCls(part='atoms', mo_basis='pm', prop='energy', verbose=0, pop_method='iao')
dip_decomp3 = decodense.DecompCls(part='atoms', mo_basis='pm', prop='dipole', verbose=0, pop_method='iao')

e_decomp4 =   decodense.DecompCls(part='eda', mo_basis='can', prop='energy', verbose=0, pop_method='iao')
dip_decomp4 = decodense.DecompCls(part='eda', mo_basis='can', prop='dipole', verbose=0, pop_method='iao')
e_decomp5 =   decodense.DecompCls(part='eda', mo_basis='fb', prop='energy', verbose=0, pop_method='iao')
dip_decomp5 = decodense.DecompCls(part='eda', mo_basis='fb', prop='dipole', verbose=0, pop_method='iao')
e_decomp6 =   decodense.DecompCls(part='eda', mo_basis='pm', prop='energy', verbose=0, pop_method='iao')
dip_decomp6 = decodense.DecompCls(part='eda', mo_basis='pm', prop='dipole', verbose=0, pop_method='iao')

e_decomp7 =   decodense.DecompCls(part='orbitals', mo_basis='can', prop='energy', verbose=0, pop_method='iao')
dip_decomp7 = decodense.DecompCls(part='orbitals', mo_basis='can', prop='dipole', verbose=0, pop_method='iao')
e_decomp8 =   decodense.DecompCls(part='orbitals', mo_basis='fb', prop='energy', verbose=0, pop_method='iao')
dip_decomp8 = decodense.DecompCls(part='orbitals', mo_basis='fb', prop='dipole', verbose=0, pop_method='iao')
e_decomp9 =   decodense.DecompCls(part='orbitals', mo_basis='pm', prop='energy', verbose=0, pop_method='iao')
dip_decomp9 = decodense.DecompCls(part='orbitals', mo_basis='pm', prop='dipole', verbose=0, pop_method='iao')

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

dip1_1 = dipole1(E0,e_decomp1,mol)
print('JAX CAN/IAO ATOMS Dipole',jnp.sum(dip1_1,0))
dip2_1 = dipole1(E0,e_decomp2,mol)
print('JAX FB/IAO ATOMS Dipole',jnp.sum(dip2_1,0))
dip3_1 = dipole1(E0,e_decomp3,mol)
print('JAX PM/IAO ATOMS Dipole',jnp.sum(dip3_1,0))
dip4_1 = dipole1(E0,e_decomp4,mol)
print('JAX CAN/IAO EDA Dipole',jnp.sum(dip4_1,0))
dip5_1 = dipole1(E0,e_decomp5,mol)
print('JAX FB/IAO EDA Dipole',jnp.sum(dip5_1,0))
dip6_1 = dipole1(E0,e_decomp6,mol)
print('JAX PM/IAO EDA Dipole',jnp.sum(dip6_1,0))
dip7_1 = dipole1(E0,e_decomp7,mol)
print('JAX CAN/IAO orbitals Dipole',jnp.sum(dip7_1,0))
dip8_1 = dipole1(E0,e_decomp8,mol)
print('JAX FB/IAO orbitals Dipole',jnp.sum(dip8_1,0))
dip9_1 = dipole1(E0,e_decomp9,mol)
print('JAX PM/IAO orbitals Dipole',jnp.sum(dip9_1,0))



dip1_2 = dipole2(E0,dip_decomp1,mol)
print('Decodense CAN/IAO ATOMS Dipole',jnp.sum(dip1_2,0))
dip2_2 = dipole2(E0,dip_decomp2,mol)
print('Decodense FB/IAO ATOMS Dipole',jnp.sum(dip2_2,0))
dip3_2 = dipole2(E0,dip_decomp3,mol)
print('Decodense PM/IAO ATOMS Dipole',jnp.sum(dip3_2,0))
dip4_2 = dipole2(E0,dip_decomp4,mol)
print('Decodense CAN/IAO EDA Dipole',jnp.sum(dip4_2,0))
dip5_2 = dipole2(E0,dip_decomp5,mol)
print('Decodense FB/IAO EDA Dipole',jnp.sum(dip5_2,0))
dip6_2 = dipole2(E0,dip_decomp6,mol)
print('Decodense PM/IAO EDA Dipole',jnp.sum(dip6_2,0))
dip7_2 = dipole2(E0,dip_decomp7,mol)
print('Decodense CAN/IAO orbitals Dipole',jnp.sum(dip7_2,0))
dip8_2 = dipole2(E0,dip_decomp8,mol)
print('Decodense FB/IAO orbitals Dipole',jnp.sum(dip8_2,0))
dip9_2 = dipole2(E0,dip_decomp9,mol)
print('Decodense PM/IAO orbitals Dipole',jnp.sum(dip9_2,0))


pol1_1 = jax.jacrev(dipole1)(E0,e_decomp1,mol)
pol2_1 = jax.jacrev(dipole1)(E0,e_decomp2,mol)
pol3_1 = jax.jacrev(dipole1)(E0,e_decomp3,mol)
pol4_1 = jax.jacrev(dipole1)(E0,e_decomp4,mol)
pol5_1 = jax.jacrev(dipole1)(E0,e_decomp5,mol)
pol6_1 = jax.jacrev(dipole1)(E0,e_decomp6,mol)
pol7_1 = jax.jacrev(dipole1)(E0,e_decomp7,mol)
pol8_1 = jax.jacrev(dipole1)(E0,e_decomp8,mol)
pol9_1 = jax.jacrev(dipole1)(E0,e_decomp9,mol)


pol1_2 = jax.jacrev(dipole2)(E0,dip_decomp1,mol)
pol2_2 = jax.jacrev(dipole2)(E0,dip_decomp2,mol)
pol3_2 = jax.jacrev(dipole2)(E0,dip_decomp3,mol)
pol4_2 = jax.jacrev(dipole2)(E0,dip_decomp4,mol)
pol5_2 = jax.jacrev(dipole2)(E0,dip_decomp5,mol)
pol6_2 = jax.jacrev(dipole2)(E0,dip_decomp6,mol)
pol7_2 = jax.jacrev(dipole2)(E0,dip_decomp7,mol)
pol8_2 = jax.jacrev(dipole2)(E0,dip_decomp8,mol)
pol9_2 = jax.jacrev(dipole2)(E0,dip_decomp9,mol)


print('1st derivative CAN/IAO Polarizability',jnp.trace(jnp.sum(pol1_2,0))/3)
print('1st derivative CAN/IAO Polarizability',jnp.trace(jnp.sum(pol2_2,0))/3)
print('1st derivative CAN/IAO Polarizability',jnp.trace(jnp.sum(pol3_2,0))/3)
print('1st derivative CAN/IAO Polarizability',jnp.trace(jnp.sum(pol4_2,0))/3)
print('1st derivative CAN/IAO Polarizability',jnp.trace(jnp.sum(pol5_2,0))/3)
print('1st derivative CAN/IAO Polarizability',jnp.trace(jnp.sum(pol6_2,0))/3)
print('1st derivative CAN/IAO Polarizability',jnp.trace(jnp.sum(pol7_2,0))/3)
print('1st derivative CAN/IAO Polarizability',jnp.trace(jnp.sum(pol8_2,0))/3)
print('1st derivative CAN/IAO Polarizability',jnp.trace(jnp.sum(pol9_2,0))/3)


print('2nd derivative CAN/IAO Polarizability',jnp.trace(jnp.sum(pol1_1,0))/3)
print('2nd derivative CAN/IAO Polarizability',jnp.trace(jnp.sum(pol2_1,0))/3)
print('2nd derivative CAN/IAO Polarizability',jnp.trace(jnp.sum(pol3_1,0))/3)
print('2nd derivative CAN/IAO Polarizability',jnp.trace(jnp.sum(pol4_1,0))/3)
print('2nd derivative CAN/IAO Polarizability',jnp.trace(jnp.sum(pol5_1,0))/3)
print('2nd derivative CAN/IAO Polarizability',jnp.trace(jnp.sum(pol6_1,0))/3)
print('2nd derivative CAN/IAO Polarizability',jnp.trace(jnp.sum(pol7_1,0))/3)
print('2nd derivative CAN/IAO Polarizability',jnp.trace(jnp.sum(pol8_1,0))/3)
print('2nd derivative CAN/IAO Polarizability',jnp.trace(jnp.sum(pol9_1,0))/3)