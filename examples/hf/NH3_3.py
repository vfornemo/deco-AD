import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import jax
from pyscfad.lib import numpy as jnp
from pyscfad import gto
import decodense
from decoAD.hf.dipole import dipole1, dipole2


# Criteria of decomposition
e_decomp1 = decodense.DecompCls(part='atoms', mo_basis='pm', prop='energy', verbose=0, pop_method='mulliken')
dip_decomp1 = decodense.DecompCls(part='atoms', mo_basis='pm', prop='dipole', verbose=0, pop_method='mulliken')
e_decomp2 = decodense.DecompCls(part='atoms', mo_basis='pm', prop='energy', verbose=0, pop_method='iao')
dip_decomp2 = decodense.DecompCls(part='atoms', mo_basis='pm', prop='dipole', verbose=0, pop_method='iao')
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

mol.basis = 'cc-pvdz'
mol.build(trace_exp=False, trace_ctr_coeff=False)

print('\n###### NH3 ######\n')

dip1_pm = dipole1(E0,e_decomp1,mol)
dip1_can = dipole1(E0,e_decomp2,mol)
print(f'JAX {e_decomp1.mo_basis}/{e_decomp1.pop_method} Dipole',jnp.sum(dip1_pm,0))
print(f'JAX {e_decomp2.mo_basis}/{e_decomp2.pop_method} Dipole',jnp.sum(dip1_can,0))


dip2_pm = dipole2(E0,dip_decomp1,mol)
dip2_can = dipole2(E0,dip_decomp2,mol)
print(f'Decodense {e_decomp1.mo_basis}/{e_decomp1.pop_method} Dipole',jnp.sum(dip2_pm,0))
print(f'Decodense {e_decomp2.mo_basis}/{e_decomp2.pop_method} Dipole',jnp.sum(dip2_can,0))

pol1_pm = jax.jacrev(dipole1)(E0,e_decomp1,mol)
pol2_pm = jax.jacrev(dipole2)(E0,dip_decomp1,mol)

pol1_can = jax.jacrev(dipole1)(E0,e_decomp2,mol)
pol2_can = jax.jacrev(dipole2)(E0,dip_decomp2,mol)

pol1_can_N = jnp.trace(pol1_can[0])
pol1_can_H1 = jnp.trace(pol1_can[1])
pol1_can_H2 = jnp.trace(pol1_can[2])
pol1_can_H3 = jnp.trace(pol1_can[3])
pol1_pm_N = jnp.trace(pol1_pm[0])
pol1_pm_H1 = jnp.trace(pol1_pm[1])
pol1_pm_H2 = jnp.trace(pol1_pm[2])
pol1_pm_H3 = jnp.trace(pol1_pm[3])
pol2_can_N = jnp.trace(pol2_can[0])
pol2_can_H1 = jnp.trace(pol2_can[1])
pol2_can_H2 = jnp.trace(pol2_can[2])
pol2_can_H3 = jnp.trace(pol2_can[3])
pol2_pm_N = jnp.trace(pol2_pm[0])
pol2_pm_H1 = jnp.trace(pol2_pm[1])
pol2_pm_H2 = jnp.trace(pol2_pm[2])
pol2_pm_H3 = jnp.trace(pol2_pm[3])

print(f'1st derivative {e_decomp2.mo_basis}/{e_decomp2.pop_method} Polarizability\n')
print('N',pol2_can_N,'H1',pol2_can_H1,'H2',pol2_can_H2,'H3',pol2_can_H3)
print(f'\n1st derivative {e_decomp1.mo_basis}/{e_decomp1.pop_method} Polarizability\n')
print('N',pol2_pm_N,'H1',pol2_pm_H1,'H2',pol2_pm_H2,'H3',pol2_pm_H3)
print(f'\n2nd derivative {e_decomp2.mo_basis}/{e_decomp2.pop_method} Polarizability\n')
print('N',pol1_can_N,'H1',pol1_can_H1,'H2',pol1_can_H2,'H3',pol1_can_H3)
print(f'\n2nd derivative {e_decomp1.mo_basis}/{e_decomp1.pop_method} Polarizability\n')
print('N',pol1_pm_N,'H1',pol1_pm_H1,'H2',pol1_pm_H2,'H3',pol1_pm_H3)
print(f'1st derivative {e_decomp2.mo_basis}/{e_decomp2.pop_method} Polarizability',jnp.trace(jnp.sum(pol2_can,0))/3)
print(f'1st derivative {e_decomp1.mo_basis}/{e_decomp1.pop_method} Polarizability',jnp.trace(jnp.sum(pol2_pm,0))/3)
print(f'2nd derivative {e_decomp2.mo_basis}/{e_decomp2.pop_method} Polarizability',jnp.trace(jnp.sum(pol1_can,0))/3)
print(f'2nd derivative {e_decomp1.mo_basis}/{e_decomp1.pop_method} Polarizability',jnp.trace(jnp.sum(pol1_pm,0))/3)