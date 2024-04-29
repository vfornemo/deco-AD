import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import jax
from pyscfad.lib import numpy as jnp
from pyscfad import gto, scf, lo, dft
import decodense

# Core Hamiltonian modified energy
def dipole1(E, decomp, mol):
    
    def dft_e(E, decomp, mf, ao_dip, h1):
        field = jnp.einsum('x,xij->ij', E, ao_dip)
        mf.get_hcore = lambda *args, **kwargs: h1 + field
        mf.kernel()
        # Localize here
        mo_coeff = (mf.mo_coeff[:, mf.mo_occ > 0.],) * 2
        mo_occ = (mf.mo_occ[mf.mo_occ > 0.] / 2.,) * 2
        ad = True
        e_part = decodense.main(mol = mol, decomp = decomp, mf = mf, mo_coeff = mo_coeff, mo_occ = mo_occ, AD = ad)
        e_tot = e_part[decodense.decomp.CompKeys.el] + e_part[decodense.decomp.CompKeys.struct]
        return e_tot
    
    def nucl_dip(E, decomp, mf, ao_dip, h1):
        field = jnp.einsum('x,xij->ij', E, ao_dip)
        mf.get_hcore = lambda *args, **kwargs: h1 + field
        mf.kernel()
        # Localize here
        mo_coeff = (mf.mo_coeff[:, mf.mo_occ > 0.],) * 2
        mo_occ = (mf.mo_occ[mf.mo_occ > 0.] / 2.,) * 2
        ad = True
        e_part = decodense.main(mol = mol, decomp = decomp, mf = mf, mo_coeff = mo_coeff, mo_occ = mo_occ, AD = ad)
        nuc_dip = e_part[decodense.decomp.CompKeys.nuc_dip]
        return nuc_dip
    
    mf = dft.RKS(mol)
    mf.conv_tol = 1e-14
    mf.kernel()
    # print("Mean Field Energy",mf.e_tot)
    ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    h1 = mf.get_hcore()
    el_dip = -jax.jacrev(dft_e)(E, decomp, mf, ao_dip, h1)
    nuc_dip = nucl_dip(E, decomp, mf, ao_dip, h1)
    #print('Electric Dipole')
    #print(el_dip)
    return el_dip + nuc_dip

def dipole2(E, decomp, mol):
    mf = dft.RKS(mol)
    mf.conv_tol = 1e-14
    mf.kernel()
    ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    h1 = mf.get_hcore()
    field = jnp.einsum('x,xij->ij', E, ao_dip)
    mf.get_hcore = lambda *args, **kwargs: h1 + field
    mf.kernel()
    # Localize here
    mo_coeff = (mf.mo_coeff[:, mf.mo_occ > 0.],) * 2
    mo_occ = (mf.mo_occ[mf.mo_occ > 0.] / 2.,) * 2
    ad = True
    # dip_part = decodense.main(mol, decomp, mf, mo_coeff, mo_occ, ad)
    dip_part = decodense.main(mol = mol, decomp = decomp, mf = mf, mo_coeff = mo_coeff, mo_occ = mo_occ, AD = ad)
    dip_tot = dip_part[decodense.decomp.CompKeys.el] + dip_part[decodense.decomp.CompKeys.struct]
    return dip_tot

