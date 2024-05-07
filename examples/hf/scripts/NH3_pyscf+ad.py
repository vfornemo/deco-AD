import pyscfad, sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from pyscfad import gto, scf
import pyscf
from pyscfad.lib import numpy as jnp
from pyscfad.lo import pipek
from decodense.tools import dim

np.set_printoptions(threshold=10000)
LOC_CONV = 1.e-14

# NH3 Molecule
mol = pyscf.gto.Mole()
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
mol.build()


# NH3 Molecule
mol2 = gto.Mole()
mol2.atom = '''
N
H 1 1.008000
H 1 1.008000 2 109.47
H 1 1.008000 2 109.47 3 120
'''

# N
# H 1 1.008000
# H 1 1.008000 2 109.47
# H 1 1.008000 2 109.47 3 120

mol2.basis = 'cc-pvdz'
mol2.build()

print('\n###### NH3 ######\n')

E0 = np.array([0., 0., 0.])
mf = pyscf.scf.RHF(mol)
mf.conv_tol = 1e-14
mf.kernel()

mf2 = scf.RHF(mol2)
mf2.conv_tol = 1e-14
mf2.kernel()

# molecular dimensions
alpha, beta = dim(mf.mo_occ)

# print("alpha", alpha)
# print("beta", beta)

print("mo_coeff", mf.mo_coeff)
print("shape of mo_coeff", mf.mo_coeff.shape)
print("type of mo_coeff", type(mf.mo_coeff))

# # loop over spins
# for i, spin_mo in enumerate((alpha, beta)):
#     # construct start guess
#     # canonical MOs as start guess
#     mo_coeff_init = mf.mo_coeff[i][:, spin_mo]

mo_coeff_out = (jnp.zeros_like(mf.mo_coeff[0]), jnp.zeros_like(mf.mo_coeff[1]))

# pipek-mezey procedure with given pop_method
# loc = lo.pipek.pm(mol, mo_coeff=mf.mo_coeff)
# loc.conv_tol = LOC_CONV
# loc.pop_method = "iao"
# loc.exponent = 2
# mo_coeff_out[i][:, spin_mo] = loc.kernel(mo_coeff_init)
# loc.kernel()

loc = pipek.pm(mol2, mf.mo_coeff[:,mf.mo_occ>0.0], pop_method = "iao", exponent = 2, conv_tol = LOC_CONV)

print("localized MO", loc)
print("type of localized MO", type(loc))


# new_array = mo_coeff_out[0].at[..., 0].set(lo.pipek.pm(mol, mf.mo_coeff, pop_method = "iao", \
#                                                         exponent = 2, conv_tol = LOC_CONV))
# mo_coeff_out = mo_coeff_out[:0] + (new_array,) + mo_coeff_out[0+1:]

# print("localized MO", mo_coeff_out)

