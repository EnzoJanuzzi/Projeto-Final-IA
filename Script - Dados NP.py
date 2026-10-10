import os
import glob
import numpy as np
import pandas as pd
from ase.io import read
from pymatgen.io.vasp import Outcar, Vasprun

def extract_np_datapoint(calc_dir):
    """
    Extracts structural, electronic, and magnetic features from a VASP run directory.
    """
    outcar_path = os.path.join(calc_dir, "OUTCAR")
    contcar_path = os.path.join(calc_dir, "CONTCAR")
    vasprun_path = os.path.join(calc_dir, "vasprun.xml")
    
    # 1. Structural features via ASE
    atoms = read(contcar_path)
    n_atoms = len(atoms)
    
    # Radius of gyration as a size metric
    com = atoms.get_center_of_mass()
    dists = np.linalg.norm(atoms.positions - com, axis=1)
    r_gyration = np.sqrt(np.mean(dists**2))
    
    # 2. Energetics & Magnetization via Pymatgen
    outcar = Outcar(outcar_path)
    vasprun = Vasprun(vasprun_path, parse_dos=True)
    
    total_energy = outcar.final_energy
    energy_per_atom = total_energy / n_atoms
    
    # Total and atomic magnetization
    total_mag = outcar.total_mag if outcar.total_mag is not None else 0.0
    site_mags = [m['tot'] for m in outcar.magnetization] if outcar.magnetization else [0.0] * n_atoms
    
    # Separate core vs surface (simple geometric criterion based on distance from center)
    cutoff_dist = np.percentile(dists, 60)
    surface_mask = dists > cutoff_dist
    
    mean_surface_mag = np.mean(np.array(site_mags)[surface_mask]) if any(surface_mask) else 0.0
    mean_core_mag = np.mean(np.array(site_mags)[~surface_mask]) if any(~surface_mask) else 0.0
    
    # 3. Electronic properties (Fermi level and DOS)
    efermi = vasprun.efermi
    
    # Compute d-band center if projected DOS exists
    cdos = vasprun.complete_dos
    element = atoms.get_chemical_symbols()[0] # Adjust for alloys
    try:
        spd_dos = cdos.get_element_spd_dos(element)
        d_dos = spd_dos[2] # 2 corresponds to d orbital
        energies = d_dos.energies - efermi
        densities = d_dos.densities[1] # spin up (or average with spin down)
        
        # Integrate below Fermi level (E <= 0)
        mask = energies <= 0
        d_band_center = np.trapz(energies[mask] * densities[mask], energies[mask]) / np.trapz(densities[mask], energies[mask])
    except Exception:
        d_band_center = np.nan

    return {
        "n_atoms": n_atoms,
        "r_gyration": r_gyration,
        "energy_per_atom": energy_per_atom,
        "total_mag": total_mag,
        "surface_mag_mean": mean_surface_mag,
        "core_mag_mean": mean_core_mag,
        "e_fermi": efermi,
        "d_band_center": d_band_center,
    }