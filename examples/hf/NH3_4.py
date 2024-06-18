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
N
H 1 1.008000
H 1 1.008000 2 109.47
H 1 1.008000 2 109.47 3 120
'''

# N
# H 1 1.008000
# H 1 1.008000 2 109.47
# H 1 1.008000 2 109.47 3 120

# mol.basis = '6-311++G**'
mol.basis = 'aug-pcseg-1'
# mol.basis = 'cc-pvdz'
mol.build(trace_exp=False, trace_ctr_coeff=False)

print('\n###### NH3 ######\n')


print("execute of dip2_mul")
dip2_mul = dipole2(E0,dip_decomp1,mol)
print(f'Decodense {dip_decomp1.mo_basis}/{dip_decomp1.pop_method} Dipole',jnp.sum(dip2_mul,0))

print("execute of dip2_iao")
dip2_iao = dipole2(E0,dip_decomp2,mol)
print(f'Decodense {dip_decomp2.mo_basis}/{dip_decomp2.pop_method} Dipole',jnp.sum(dip2_iao,0))


print("jax rev of pol2_mul")
pol2_mul = jax.jacrev(dipole2)(E0,dip_decomp1,mol)

print("jax rev of pol2_iao")
pol2_iao = jax.jacrev(dipole2)(E0,dip_decomp2,mol)


pol2_iao_N = jnp.trace(pol2_iao[0])/ 3
pol2_iao_H1 = jnp.trace(pol2_iao[1])/ 3
pol2_iao_H2 = jnp.trace(pol2_iao[2])/ 3
pol2_iao_H3 = jnp.trace(pol2_iao[3])/ 3
pol2_mul_N = jnp.trace(pol2_mul[0])/ 3
pol2_mul_H1 = jnp.trace(pol2_mul[1])/ 3
pol2_mul_H2 = jnp.trace(pol2_mul[2])/ 3
pol2_mul_H3 = jnp.trace(pol2_mul[3])/ 3

print(f'1st derivative {dip_decomp2.mo_basis}/{dip_decomp2.pop_method} Polarizability\n')
print('N',pol2_iao_N,'H1',pol2_iao_H1,'H2',pol2_iao_H2,'H3',pol2_iao_H3)
print(f'\n1st derivative {dip_decomp1.mo_basis}/{dip_decomp1.pop_method} Polarizability\n')
print('N',pol2_mul_N,'H1',pol2_mul_H1,'H2',pol2_mul_H2,'H3',pol2_mul_H3)
print(f'1st derivative {dip_decomp2.mo_basis}/{dip_decomp2.pop_method} Polarizability',jnp.trace(jnp.sum(pol2_iao,0))/3)
print(f'1st derivative {dip_decomp1.mo_basis}/{dip_decomp1.pop_method} Polarizability',jnp.trace(jnp.sum(pol2_mul,0))/3)