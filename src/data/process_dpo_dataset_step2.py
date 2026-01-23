import argparse
from pathlib import Path
import numpy as np
import torch
import shutil
import logging
import sys

# Setup imports similar to process_dpo_dataset.py
basedir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(basedir))
from src import utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dpo-criterion', type=str, default='reos.all', 
                        choices=['reos.all', 'medchem.sa', 'medchem.qed', 'gnina.vina_efficiency','enamine.avail','combined','freedom_space.max_similarity'])
    parser.add_argument('--basedir', type=Path, default=None)
    parser.add_argument('--pocket', type=str, default='CA+',
                        choices=['side_chain_bead', 'CA+'])
    parser.add_argument('--flex', action='store_true')
    parser.add_argument('--normal_modes', action='store_true')
    parser.add_argument('--toy', action='store_true')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of jobs to merge')
    parser.add_argument('--datadir', type=Path, required=True, help='Original data directory for copying stats/val/test')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    utils.setup_logging()

    # Reconstruct output directory name
    dirname = f"dpo_{args.dpo_criterion.replace('.','_')}_{args.pocket}"
    if args.flex:
        dirname += '_flex'
    if args.normal_modes:
        dirname += '_nma'
    if args.toy:
        dirname += '_toy'
    processed_dir = Path(args.basedir, dirname)
    chunks_dir = processed_dir / 'chunks'
    
    logging.info(f"Merging results from {chunks_dir} into {processed_dir}")

    # Initialize containers
    merged_data = {
        'ligands_w': {},
        'ligands_l': {},
        'pockets': {}
    }
    
    merged_smiles = []
    
    # Iterate over jobs
    for job_id in range(args.n_jobs):
        logging.info(f"Processing job {job_id}...")
        
        # Load train PT
        pt_path = chunks_dir / f'train_{job_id}.pt'
        if not pt_path.exists():
            logging.warning(f"File {pt_path} does not exist. Skipping job {job_id}.")
            continue
            
        try:
            data = torch.load(pt_path)
            
            # Merge dictionaries
            for cat in ['ligands_w', 'ligands_l', 'pockets']:
                for key, val in data[cat].items():
                    if key not in merged_data[cat]:
                        merged_data[cat][key] = []
                    merged_data[cat][key].extend(val)
                        
            # Load smiles
            smiles_path = chunks_dir / f'train_smiles_{job_id}.npy'
            if smiles_path.exists():
                smiles = np.load(smiles_path)
                merged_smiles.extend(smiles)
            else:
                logging.warning(f"Smiles file {smiles_path} not found.")
                
        except Exception as e:
            logging.error(f"Error processing job {job_id}: {e}")

    # Save merged data
    logging.info(f"Saving merged train.pt with {len(merged_smiles)//2} pairs")
    torch.save(merged_data, processed_dir / 'train.pt')
    
    logging.info(f"Saving merged train_smiles.npy")
    np.save(processed_dir / 'train_smiles.npy', np.array(merged_smiles))

    # Copy stats and val/test sets from datadir
    logging.info(f"Copying stats and validation/test sets from {args.datadir}")
    
    files_to_copy = [
        'size_distribution.npy',
        'ligand_type_histogram.npy',
        'ligand_bond_type_histogram.npy',
        'metadata.yml',
        'val.pt',
        'test.pt'
    ]
    
    for fname in files_to_copy:
        src = args.datadir / fname
        if src.exists():
            shutil.copy(src, processed_dir)
        else:
            logging.warning(f"Source file {src} not found")

    dirs_to_copy = ['val', 'test']
    for dname in dirs_to_copy:
        src = args.datadir / dname
        dst = processed_dir / dname
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
             logging.warning(f"Source dir {src} not found")

    # Merge errors
    all_errors = ""
    for job_id in range(args.n_jobs):
        err_path = chunks_dir / f'errors_{job_id}.txt'
        if err_path.exists():
            err_text = err_path.read_text()
            if err_text.strip():
                all_errors += f"\n--- Errors from Job {job_id} ---\n"
                all_errors += err_text
    
    with open(processed_dir / 'errors.txt', 'w') as f:
        f.write(all_errors)
            
    logging.info("Merge complete.")

if __name__ == '__main__':
    main()
