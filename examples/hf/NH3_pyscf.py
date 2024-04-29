import pyscf, sys
import numpy as np
from pyscf import gto, lo, scf, tools
from pyscf.lo import pipek, iao, orth, boys

np.set_printoptions(threshold=40000)
LOC_CONV = 1.e-10

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
mol.build()

print('\n###### NH3 ######\n')

E0 = np.array([0., 0., 0.])
mf = scf.RHF(mol)
mf.conv_tol = 1e-14
mf.kernel()

print("mo_coeff", mf.mo_coeff)
print("type of mo_coeff", type(mf.mo_coeff))


# ao_dip = mol.intor_symmetric('int1e_r', comp=3)
# h1 = mf.get_hcore()
# field = np.einsum('x,xij->ij', E0, ao_dip)
# mf.get_hcore = lambda *args, **kwargs: h1 + field
# mf.kernel()

# pipek-mezey procedure with given pop_method
loc = lo.PM(mol, mf=mf, mo_coeff=mf.mo_coeff[:,mf.mo_occ>0.0])
loc.conv_tol = LOC_CONV
loc.pop_method = "iao"
loc.exponent = 2
# mo_coeff_out = (np.zeros_like(mo_coeff_in[0]), np.zeros_like(mo_coeff_in[1]))
# mo_coeff_out[i][:, spin_mo] = loc.kernel(mo_coeff_init)
loc.kernel()

print("localized MO", loc.mo_coeff)

