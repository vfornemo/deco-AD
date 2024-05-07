import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import jax
from pyscfad.lib import numpy as jnp
from pyscfad import gto
import decodense
from decoAD.hf.dipole import dipole1, dipole2
import pandas as pd

jnp.set_printoptions(threshold=100000)
jnp.set_printoptions(linewidth=jnp.inf)

# print this script
print(open(__file__).read())
print("-------------- Log starts here --------------")

nh3 = gto.Mole()
nh3.atom = '''
N
H 1 1.008000
H 1 1.008000 2 109.47
H 1 1.008000 2 109.47 3 120
'''
hf = gto.Mole()
hf.atom = '''
H    0.0000   0.0000    0.0000 
F    0.0000   0.0000    0.9170
'''

h2o = gto.Mole()
h2o.atom = '''
O    0.0000    0.0000    0.1141
H    0.0000    0.7804   -0.4563
H    0.0000   -0.7804   -0.4563
'''

ch4 = gto.Mole()
ch4.atom = '''
C    0.0000   -0.0000    0.0000 
H    0.0000   -0.8900   -0.6293 
H    0.0000    0.8900   -0.6293 
H   -0.8900   -0.0000    0.6293 
H    0.8900   -0.0000    0.6293 
'''

MOL = [nh3, h2o, hf, ch4]
MOL2 = ['NH3', 'H2O', 'HF', 'CH4']
MOL_DICT = {'NH3': nh3, 'H2O': h2o, 'HF': hf, 'CH4': ch4}
BASIS = ['aug-pcseg-1', '6-311++G**']
MO = ['pm', 'can', 'fb']
POP = ['mulliken','iao']

# save result to csv
filename = 'bulk_test1.csv'
f = open(filename, 'a', buffering = 1)

f.write("count,molecule,basis set,mo_basis,pop_method,dipJAX,dipDeco,polar1st,polar2nd\n")

E0 = jnp.array([0., 0., 0.])

    
for mol2 in MOL2:
    for basis in BASIS:
        for mo in MO:
            for pop in POP:
                for i in range(5):
                    if mol2 != 'NH3':
                        print(f'{mol2} {basis} {mo} {pop} {i}\n')
                        mol = MOL_DICT[mol2]
                        # Criteria of decomposition
                        e_decomp1 = decodense.DecompCls(part='atoms', mo_basis=mo, prop='energy', verbose=0, pop_method=pop)
                        dip_decomp1 = decodense.DecompCls(part='atoms', mo_basis=mo, prop='dipole', verbose=0, pop_method=pop)
                        # Static external electric field
                        mol.basis = basis
                        # mol.build()
                        mol.build(trace_exp=False, trace_ctr_coeff=False)

                        print("execute of dip1_mul")
                        dip1_mul = dipole1(E0,e_decomp1,mol)
                        print(f'JAX {e_decomp1.mo_basis}/{e_decomp1.pop_method} Dipole',jnp.sum(dip1_mul,0))

                        print("execute of dip2_mul")
                        dip2_mul = dipole2(E0,dip_decomp1,mol)
                        print(f'Decodense {e_decomp1.mo_basis}/{e_decomp1.pop_method} Dipole',jnp.sum(dip2_mul,0))

                        print("jax rev of pol1_mul")
                        pol1_mul = jax.jacrev(dipole1)(E0,e_decomp1,mol)

                        print("jax rev of pol2_mul")
                        pol2_mul = jax.jacrev(dipole2)(E0,dip_decomp1,mol)

                        print(f'1st derivative {e_decomp1.mo_basis}/{e_decomp1.pop_method} Polarizability',jnp.trace(jnp.sum(pol2_mul,0))/3)
                        print(f'2nd derivative {e_decomp1.mo_basis}/{e_decomp1.pop_method} Polarizability',jnp.trace(jnp.sum(pol1_mul,0))/3)

                        f.write(f'{i},{mol2},{basis},{mo},{pop},{jnp.sum(dip1_mul,0)},{jnp.sum(dip2_mul,0)},{jnp.trace(jnp.sum(pol2_mul,0))/3},{jnp.trace(jnp.sum(pol1_mul,0))/3}\n')
                
f.close()