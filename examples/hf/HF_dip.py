import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import jax
from pyscfad.lib import numpy as jnp
from pyscfad import gto
import decodense
from decoAD.hf.dipole import dipole1, dipole2
from pyscfad import config

config.update('pyscfad_scf_implicit_diff', True)

jnp.set_printoptions(threshold=100000)
jnp.set_printoptions(linewidth=jnp.inf)

# print this script
print(open(__file__).read())
print("-------------- Log starts here --------------")

# Criteria of decomposition
dip_decomp1 = decodense.DecompCls(part='atoms', mo_basis='pm', prop='dipole', verbose=0, pop_method='mulliken')
dip_decomp2 = decodense.DecompCls(part='atoms', mo_basis='pm', prop='dipole', verbose=0, pop_method='iao')
# Static external electric field
E0 = jnp.array([0., 0., 0.])

# NH3 Molecule
mol = gto.Mole()
mol.atom = '''
H
F 1 0.91
'''

# mol.basis = '6-311++G**'
mol.basis = 'aug-pcseg-1'
# mol.basis = 'cc-pvdz'
mol.build(trace_coords=False, trace_exp=False, trace_ctr_coeff=False)


print('\n###### NH3 ######\n')


dip2_pm = dipole2(E0,dip_decomp1,mol)
dip2_can = dipole2(E0,dip_decomp2,mol)
print(f'Decodense {dip_decomp1.mo_basis}/{dip_decomp1.pop_method} Dipole',jnp.sum(dip2_pm,0))
print(f'Decodense {dip_decomp2.mo_basis}/{dip_decomp2.pop_method} Dipole',jnp.sum(dip2_can,0))

pol2_pm = jax.jacrev(dipole2)(E0,dip_decomp1,mol)

pol2_can = jax.jacrev(dipole2)(E0,dip_decomp2,mol)


pol2_can_H = jnp.trace(pol2_can[0])/ 3
pol2_can_F = jnp.trace(pol2_can[1])/ 3


pol2_pm_H = jnp.trace(pol2_pm[0])/ 3
pol2_pm_F = jnp.trace(pol2_pm[1])/ 3


print(f'1st derivative {dip_decomp2.mo_basis}/{dip_decomp2.pop_method} Polarizability\n')
print('H',pol2_can_H,'F',pol2_can_F)
print(f'\n1st derivative {dip_decomp1.mo_basis}/{dip_decomp1.pop_method} Polarizability\n')
print('H',pol2_pm_H,'F',pol2_pm_F)

print(f'1st derivative {dip_decomp2.mo_basis}/{dip_decomp2.pop_method} Polarizability',jnp.trace(jnp.sum(pol2_can,0))/3)
print(f'1st derivative {dip_decomp1.mo_basis}/{dip_decomp1.pop_method} Polarizability',jnp.trace(jnp.sum(pol2_pm,0))/3)