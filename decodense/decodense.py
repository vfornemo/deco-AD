#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
main mf_decomp program
"""

__author__ = 'Janus Juul Eriksen, Technical University of Denmark, DK'
__maintainer__ = 'Janus Juul Eriksen'
__email__ = 'janus@kemi.dtu.dk'
__status__ = 'Development'

import numpy as np
from pyscfad.lib import numpy as jnp
import pandas as pd
from pyscfad import gto, scf, dft
from typing import Dict, Tuple, List, Union, Optional, Any

from .decomp import DecompCls, sanity_check
from .orbitals import loc_orbs, assign_rdm1s
from .properties import prop_tot
from .tools import make_natorb, mf_info, write_rdm1
from .results import fmt


def main(mol: gto.Mole, decomp: DecompCls, \
         mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT], \
         mo_coeff: jnp.ndarray, # = None, \
         mo_occ: jnp.ndarray, # = None,
         rdm1_orb: jnp.ndarray = None, \
         rdm1_eff: jnp.ndarray = None, AD = False) -> pd.DataFrame:
        """
        main decodense program
        """
        # sanity check
        sanity_check(mol, decomp)

        # get orbitals and mo occupation
        # if mo_coeff is None or mo_occ is None:
        #     # format orbitals from mean-field calculation
        #     if rdm1_orb is None:
        #         mo_coeff, mo_occ = mf_info(mf)
        #         # print("mo_coeff", mo_coeff)
        #         # print("mo_occ", mo_occ)
        #     else:
        #         mo_coeff, mo_occ = make_natorb(mol, jnp.asarray(mf.mo_coeff), jnp.asarray(rdm1_orb))
        #     # compute localized MOs
        #     if decomp.mo_basis != 'can':
        #         mo_coeff = loc_orbs(mol, mf, mo_coeff, mo_occ, \
        #                             decomp.mo_basis, decomp.pop_method, decomp.mo_init, decomp.loc_exp, \
        #                             decomp.ndo, decomp.verbose)
                
        # print("new mo_coeff", mo_coeff)

        # compute population weights
        # weights = assign_rdm1s(mol, mf, mo_coeff, mo_occ, decomp.pop_method, decomp.part, \
        #                        decomp.ndo, decomp.verbose)
        
        # print("weights", weights)
        
        # weights = jnp.ones((len(mo_occ[0]),mol.natm))
        # weights = weights * 0.25
        
        # weights = jnp.asarray([[ 0.25, 0.25, 0.25, 0.25],
        #            [ 0.25, 0.25, 0.25, 0.25],
        #            [ 0.25, 0.25, 0.25, 0.25],
        #            [ 0.25, 0.25, 0.25, 0.25],
        #            [ 0.25, 0.25, 0.25, 0.25]])
        # weights = (weights,weights)
        
        w = np.random.rand(mol.natm)
        w /= w.sum()
        weights = jnp.asarray([w for i in range(len(mo_occ[0]))])
        weights = (weights,weights)
        print("weights", weights)
        
        # compute decomposed results
        decomp.res = prop_tot(mol, mf, mo_coeff, mo_occ, rdm1_eff, \
                              decomp.pop_method, decomp.prop, decomp.part, \
                              decomp.ndo, decomp.gauge_origin, weights, AD)

        # write rdm1s
        if decomp.write != '':
            write_rdm1(mol, decomp.part, mo_coeff, mo_occ, decomp.write, weights)
        
        if AD == True:
            return decomp.res
        else:
            return fmt(mol, decomp.res, decomp.unit, decomp.ndo)

