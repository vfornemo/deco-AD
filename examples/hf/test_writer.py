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
POP = ['mulliken', 'iao']

# save result to csv
filename = 'bulk_test.csv'
f = open(filename, 'w')

f.write("count,molecule,basis set,mo_basis,pop_method,dipJAX,dipDeco,polar1st,polar2nd\n")

E0 = jnp.array([0., 0., 0.])

    
for mol2 in MOL2:
    for basis in BASIS:
        for mo in MO:
            for pop in POP:
                for i in range(5):
                    print(f'{mol2} {basis} {mo} {pop}\n')
                    mol = MOL_DICT[mol2]
                    f.write(f'{i},{mol2},{basis},{mo},{pop}\n')
                
f.close()