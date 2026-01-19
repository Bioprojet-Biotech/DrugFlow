import argparse
from pathlib import Path
import shutil
from time import time
from Bio.PDB import PDBParser
from rdkit import Chem
import torch
import logging
from collections import defaultdict
from tqdm import tqdm

import sys
basedir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(basedir))

from src.data.data_utils import process_raw_pair, rdmol_to_smiles

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ligand', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--datadir', type=Path, required=True)
    parser.add_argument('--basedir', type=Path, default=None)
    parser.add_argument('--pocket', type=str, default='CA+',
                        choices=['side_chain_bead', 'CA+'])
    parser.add_argument('--normal_modes', action='store_true')
    parser.add_argument('--flex', action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    assert args.datadir.exists(), f"Data directory {args.datadir} does not exist."
    assert args.ligand.exists(), f"Ligand file {args.ligand} does not exist."
    assert args.target.exists(), f"Target file {args.target} does not exist."
    
    # Make output directory
    dirname = f"single_pocket_{args.pocket}"
    if args.flex:
        dirname += '_flex'
    if args.normal_modes:
        dirname += '_nma'
    processed_dir = Path(args.basedir, dirname)
    processed_dir.mkdir(parents=True, exist_ok=True)

    failed = {}
    # dataset has only one pocket-ligand pair
    datasets = {
        'train': [(args.target, args.ligand)],
        'val': [(args.target, args.ligand)],
        'test': [(args.target, args.ligand)]
    }
    for split in ['train', 'val', 'test']:

        print(f"Processing {split} dataset...")

        ligands = defaultdict(list)
        pockets = defaultdict(list)

        pbar = tqdm(datasets[split])
        for pocket_fn, ligand_fn in pbar:


            sdffile = ligand_fn
            pdbfile = pocket_fn

            try:
                pdb_model = PDBParser(QUIET=True).get_structure('', pdbfile)[0]

                rdmol = Chem.SDMolSupplier(str(sdffile))[0]

                ligand, pocket = process_raw_pair(
                    pdb_model, rdmol, pocket_representation=args.pocket,
                    compute_nerf_params=args.flex, compute_bb_frames=args.flex,
                    nma_input=pdbfile if args.normal_modes else None)

            except (KeyError, AssertionError, FileNotFoundError, IndexError,
                    ValueError, AttributeError) as e:
                failed[(split, sdffile, pdbfile)] = (type(e).__name__, str(e))
                continue

            nerf_keys = ['fixed_coord', 'atom_mask', 'nerf_indices', 'length', 'theta', 'chi', 'ddihedral', 'chi_indices']
            for k in ['x', 'one_hot', 'bonds', 'bond_one_hot', 'v', 'nma_vec'] + nerf_keys + ['axis_angle']:
                if k in ligand:
                    ligands[k].append(ligand[k])
                if k in pocket:
                    pockets[k].append(pocket[k])

            pocket_file = pdbfile.name.replace('_', '-')
            ligand_file = Path(pocket_file).stem + '_' + Path(sdffile).name.replace('_', '-')
            ligands['name'].append(ligand_file)
            pockets['name'].append(pocket_file)

            if split in {'val', 'test'}:
                pdb_sdf_dir = processed_dir / split
                pdb_sdf_dir.mkdir(exist_ok=True)

                # Copy PDB file
                pdb_file_out = Path(pdb_sdf_dir, pocket_file)
                shutil.copy(pdbfile, pdb_file_out)

                # Copy SDF file
                sdf_file_out = Path(pdb_sdf_dir, ligand_file)
                shutil.copy(sdffile, sdf_file_out)

        data = {'ligands': ligands, 'pockets': pockets}
        torch.save(data, Path(processed_dir, f'{split}.pt'))

    # cp stats from original dataset
    size_distr_p = Path(args.datadir, 'size_distribution.npy')
    type_histo_p = Path(args.datadir, 'ligand_type_histogram.npy')
    bond_histo_p = Path(args.datadir, 'ligand_bond_type_histogram.npy')
    metadata_p = Path(args.datadir, 'metadata.yml')
    train_smiles_p = Path(args.datadir, 'train_smiles.npy')
    shutil.copy(size_distr_p, processed_dir)
    shutil.copy(type_histo_p, processed_dir)
    shutil.copy(bond_histo_p, processed_dir)
    shutil.copy(metadata_p, processed_dir)
    shutil.copy(train_smiles_p, processed_dir)

    # Write error report
    error_str = ""
    for k, v in failed.items():
        error_str += f"{'Split':<15}:  {k[0]}\n"
        error_str += f"{'Ligand W':<15}:  {k[1]}\n"
        error_str += f"{'Ligand L':<15}:  {k[2]}\n"
        error_str += f"{'Pocket':<15}:  {k[3]}\n"
        error_str += f"{'Error type':<15}:  {v[0]}\n"
        error_str += f"{'Error msg':<15}:  {v[1]}\n\n"

    with open(Path(processed_dir, 'errors.txt'), 'w') as f:
        f.write(error_str)

    with open(Path(processed_dir, 'dataset_config.txt'), 'w') as f:
        f.write(str(args))

if __name__ == '__main__':
    main()