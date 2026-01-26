import argparse
from pathlib import Path
import numpy as np
import random
import shutil
from time import time
from collections import defaultdict
from Bio.PDB import PDBParser
from rdkit import Chem
import torch
from tqdm import tqdm
import pandas as pd
from itertools import combinations
import logging

import sys
basedir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(basedir))

from src.sbdd_metrics.metrics import REOSEvaluator, MedChemEvaluator, PoseBustersEvaluator, GninaEvalulator
from src.data.data_utils import process_raw_pair, rdmol_to_smiles
from src import utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smplsdir', type=Path, required=True)
    parser.add_argument('--metrics-detailed', type=Path, required=False)
    parser.add_argument('--ignore-missing-scores', action='store_true')
    parser.add_argument('--datadir', type=Path, required=True)
    parser.add_argument('--dpo-criterion', type=str, default='reos.all', 
                        choices=['reos.all', 'medchem.sa', 'medchem.qed', 'gnina.vina_efficiency',
                                 'enamine.avail','combined','freedom_space.max_similarity','freedom_space.avail'])
    parser.add_argument('--basedir', type=Path, default=None)
    parser.add_argument('--pocket', type=str, default='CA+',
                        choices=['side_chain_bead', 'CA+'])
    parser.add_argument('--gnina', type=Path, default='gnina')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--normal_modes', action='store_true')
    parser.add_argument('--flex', action='store_true')
    parser.add_argument('--toy', action='store_true')
    parser.add_argument('--toy_size', type=int, default=100)
    parser.add_argument('--n_pairs', type=int, default=5)
    parser.add_argument('--job_id', type=int, default=0, help='Job ID')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of jobs')
    args = parser.parse_args()
    return args

def scan_smpl_dir(samples_dir, precomp_scores=None, job_id=0, n_jobs=1):
    if precomp_scores is not None:
        logging.info("Using precomputed scores to identify valid directories")
        paths = [Path(p) for p in precomp_scores.index]
        seen = set()
        subdirs = []
        for p in paths:
            parent = p.parent
            if parent not in seen:
                seen.add(parent)
                subdirs.append(parent)
        
        subdirs = sorted(subdirs)
        if n_jobs > 1:
            chunk_size = len(subdirs) // n_jobs
            start_idx = job_id * chunk_size
            end_idx = start_idx + chunk_size if job_id < n_jobs - 1 else len(subdirs)
            subdirs = subdirs[start_idx:end_idx]

        return subdirs

    samples_dir = Path(samples_dir)
    
    all_subdirs = sorted([d for d in samples_dir.iterdir() if d.is_dir()])

    if n_jobs > 1:
        chunk_size = len(all_subdirs) // n_jobs
        start_idx = job_id * chunk_size
        end_idx = start_idx + chunk_size if job_id < n_jobs - 1 else len(all_subdirs)
        my_subdirs = all_subdirs[start_idx:end_idx]
    else:
        my_subdirs = all_subdirs

    subdirs = []
    for subdir in tqdm(my_subdirs, desc=f'Scanning samples (job {job_id})'):
        if not sample_dir_valid(subdir):
            continue
        subdirs.append(subdir)
    return subdirs

def sample_dir_valid(samples_dir):
    pocket = samples_dir / '0_pocket.pdb'
    if not pocket.exists():
        return False
    ligands = list(samples_dir.glob('*_ligand.sdf'))
    if len(ligands) < 2:
        return False
    # for ligand in ligands:
    #     if ligand.stat().st_size == 0:
    #         return False
    return True

def return_winning_losing_smpl(score_1, score_2, criterion):
    if criterion == 'reos.all':
        if score_1 == score_2:
            return None
        return score_1 > score_2
    elif criterion == 'medchem.sa':
        if np.abs(score_1 - score_2) < 0.5:
            return None
        return score_1 < score_2
    elif criterion == 'medchem.qed':
        if np.abs(score_1 - score_2) < 0.1:
            return None
        return score_1 > score_2
    elif criterion == 'gnina.vina_efficiency':
        if np.abs(score_1 - score_2) < 0.1:
            return None
        return score_1 < score_2
    elif criterion == 'enamine.avail':
        if score_1 == score_2:
            return None
        return score_1 > score_2
    elif criterion == 'freedom_space.max_similarity':
        if np.abs(score_1 - score_2) < 0.65:
            return None
        return score_1 > score_2
    elif criterion == 'freedom_space.avail':
        if max(score_1, score_2) < 0.99 or np.abs(score_1 - score_2) < 0.75:
            return None
        return score_1 > score_2
    elif criterion == 'combined':
        score_reos_1, score_reos_2 = score_1['reos.all'], score_2['reos.all']
        score_sa_1, score_sa_2 = score_1['medchem.sa'], score_2['medchem.sa']
        score_qed_1, score_qed_2 = score_1['medchem.qed'], score_2['medchem.qed']
        score_vina_1, score_vina_2 = score_1['gnina.vina_efficiency'], score_2['gnina.vina_efficiency']
        if score_reos_1 == score_reos_2: return None
        # checking consistency
        reos_sign = score_reos_1 > score_reos_2
        sa_sign = score_sa_1 < score_sa_2
        qed_sign = score_qed_1 > score_qed_2
        vina_sign = score_vina_1 < score_vina_2
        signs = [reos_sign, sa_sign, qed_sign, vina_sign]
        if all(signs) or not any(signs): return signs[0]
        return None
    
def compute_scores(sample_dirs, evaluator, criterion, n_pairs=5, toy=False, toy_size=100, 
                   precomp_scores=None, ignore_missing_scores=False):
    samples = []
    pose_evaluator = PoseBustersEvaluator()
    pbar = tqdm(sample_dirs, desc='Computing scores for samples')
    
    for dir in pbar:
        pocket = dir / '0_pocket.pdb'
        ligands = list(dir.glob('*_ligand.sdf'))
        
        target_samples = []
        for lig_path in ligands:
            mol_props = {}
            if precomp_scores is not None and str(lig_path) in precomp_scores.index:
                mol_props = precomp_scores.loc[str(lig_path)].to_dict()
                if criterion == 'combined':
                    if not 'reos.all' in mol_props or not 'medchem.sa' in mol_props or not 'medchem.qed' in mol_props or not 'gnina.vina_efficiency' in mol_props:
                        logging.debug(f'Missing combined scores for ligand: {lig_path}')
                        continue
                    mol_props['combined'] = {
                        'reos.all': mol_props['reos.all'],
                        'medchem.sa': mol_props['medchem.sa'],
                        'medchem.qed': mol_props['medchem.qed'],
                        'gnina.vina_efficiency': mol_props['gnina.vina_efficiency'],
                        'combined': mol_props['gnina.vina_efficiency']
                    }
                if 'freedom_space.avail' not in mol_props and 'freedom_space.max_similarity' in mol_props:
                    mol_props['freedom_space.avail'] = mol_props['freedom_space.max_similarity']
            
            if criterion not in mol_props and ignore_missing_scores:
                logging.debug(f'Missing {criterion} for ligand: {lig_path}')
                continue

            if 'posebusters.all' not in mol_props and ignore_missing_scores:
                logging.debug(f'Missing PoseBusters for ligand: {lig_path}')
                continue

            try:
                mol = Chem.SDMolSupplier(str(lig_path))[0]
                if mol is None:
                    continue
                smiles = rdmol_to_smiles(mol)
            except Exception as e:
                logging.error(f'Failed to read ligand: {lig_path} with error: {e}')
                continue
            
            if criterion not in mol_props:
                logging.debug(f'Recomputing {criterion} for ligand: {lig_path}')
                try:
                    eval_res = evaluator.evaluate(mol)
                    criterion_cat = criterion.split('.')[0]
                    eval_res = {f'{criterion_cat}.{k}': v for k, v in eval_res.items()}
                    score = eval_res[criterion]
                except:
                    continue
            else:
                score = mol_props[criterion]

            if 'posebusters.all' not in mol_props:
                logging.debug(f'Recomputing PoseBusters for ligand: {lig_path}')
                try:
                    pose_eval_res = pose_evaluator.evaluate(lig_path, pocket)
                except:
                    continue
                if 'all' not in pose_eval_res or not pose_eval_res['all']:
                    continue
            else:
                pose_eval_res = mol_props['posebusters.all']
                if not pose_eval_res:
                    continue
            
            if 'medchem.size' in mol_props:
                size = mol_props['medchem.size']
            else:
                try:
                    size = mol.GetNumAtoms()
                except:
                    size = 0

            if isinstance(score, (int, float, np.floating, np.integer)):
                if np.isnan(score):
                    continue

            target_samples.append({
                'smiles': smiles,
                'score': score,
                'size': size,
                'ligand_path': lig_path,
                'pocket_path': pocket
            })
        
        # Deduplicate by SMILES
        unique_samples = {}
        for sample in target_samples:
            if sample['smiles'] not in unique_samples:
                unique_samples[sample['smiles']] = sample
        unique_samples = list(unique_samples.values())
        if len(unique_samples) < 2:
            continue
        
        # Generate all possible pairs
        all_pairs = list(combinations(unique_samples, 2))
        
        # Calculate score differences and filter valid pairs
        valid_pairs = []
        for s1, s2 in all_pairs:
            sign = return_winning_losing_smpl(s1['score'], s2['score'], criterion)
            if sign is None:
                continue
            
            if criterion in ['reos.all', 'enamine.avail', 'freedom_space.avail']:
                # prioritize pairs with similar size
                score_diff = -abs(s1['size'] - s2['size'])
            elif criterion in ['freedom_space.max_similarity']:
                # bucket the similarity difference, align size
                sim_diff = abs(s1['score'] - s2['score'])
                bucket = int(sim_diff * 5)
                size_diff = abs(s1['size'] - s2['size'])
                score_diff = bucket * 1000 - size_diff + sim_diff
            else:
                score_diff = abs(s1['score'] - s2['score']) if not criterion == 'combined' else \
                            abs(s1['score']['combined'] - s2['score']['combined'])
            if sign:
                valid_pairs.append((s1, s2, score_diff))
            elif sign is False:
                valid_pairs.append((s2, s1, score_diff))
        
        # Sort pairs by score difference (descending) and select top N pairs
        valid_pairs.sort(key=lambda x: x[2], reverse=True)
        used_ligand_paths = set()
        selected_pairs = []     
        for winning, losing, score_diff in valid_pairs:
            if winning['ligand_path'] in used_ligand_paths or losing['ligand_path'] in used_ligand_paths:
                continue
            
            selected_pairs.append((winning, losing, score_diff))
            used_ligand_paths.add(winning['ligand_path'])
            used_ligand_paths.add(losing['ligand_path'])
            
            if len(selected_pairs) == n_pairs:
                break   
        for winning, losing, _ in selected_pairs:
            d = {
                'score_w': winning['score'],
                'score_l': losing['score'],
                'pocket_p': winning['pocket_path'],
                'ligand_p_w': winning['ligand_path'],
                'ligand_p_l': losing['ligand_path'],
                'size_w': winning['size'],
                'size_l': losing['size']
            }
            if isinstance(winning['score'], dict):
                for k, v in winning['score'].items():
                    d[f'{k}_w'] = v
                d['score_w'] = winning['score']['combined']
            if isinstance(losing['score'], dict):
                for k, v in losing['score'].items():
                    d[f'{k}_l'] = v
                d['score_l'] = losing['score']['combined']
            samples.append(d)                
        
        pbar.set_postfix({'added pairs': len(samples)})
        
        if toy and len(samples) >= toy_size:
            break
    
    return samples

def process_single_entry(entry, args_pocket, is_flex, normal_modes):
    try:
        pdbfile = Path(entry['pocket_p'])
        entry_ligand_p_w = Path(entry['ligand_p_w'])
        entry_ligand_p_l = Path(entry['ligand_p_l'])
        
        ligand_w_mol = Chem.SDMolSupplier(str(entry_ligand_p_w))[0]
        ligand_l_mol = Chem.SDMolSupplier(str(entry_ligand_p_l))[0]

        pdb_model = PDBParser(QUIET=True).get_structure('', pdbfile)[0]

        ligand_w_data, pocket = process_raw_pair(
            pdb_model, ligand_w_mol, pocket_representation=args_pocket,
            compute_nerf_params=is_flex, compute_bb_frames=is_flex,
            nma_input=pdbfile if normal_modes else None)
        ligand_l_data, _ = process_raw_pair(
            pdb_model, ligand_l_mol, pocket_representation=args_pocket,
            compute_nerf_params=is_flex, compute_bb_frames=is_flex,
            nma_input=pdbfile if normal_modes else None)
            
        smpl_n = pdbfile.parent.name
        
        return {
            'status': 'success',
            'ligand_w': ligand_w_data,
            'ligand_l': ligand_l_data,
            'pocket': pocket,
            'smiles_w': rdmol_to_smiles(ligand_w_mol),
            'smiles_l': rdmol_to_smiles(ligand_l_mol),
            'ligand_w_name': f'{smpl_n}__{entry_ligand_p_w.stem}.sdf',
            'ligand_l_name': f'{smpl_n}__{entry_ligand_p_l.stem}.sdf',
            'pocket_name': f'{smpl_n}__{pdbfile.stem}.pdb'
        }

    except (KeyError, AssertionError, FileNotFoundError, IndexError,
            ValueError, AttributeError) as e:
        return {
            'status': 'failed',
            'key': (entry['pocket_p'], str(entry_ligand_p_w), str(entry_ligand_p_l)),
            'error': (type(e).__name__, str(e))
        }

def main():
    args = parse_args()
    utils.setup_logging()

    logging.info(f"Processing DPO dataset with criterion: {args.dpo_criterion}")
    if 'reos' in args.dpo_criterion:
        evaluator = REOSEvaluator()
    elif 'medchem' in args.dpo_criterion:
        evaluator = MedChemEvaluator()
    elif 'gnina' in args.dpo_criterion:
        evaluator = GninaEvalulator(gnina=args.gnina)
    elif 'combined' in args.dpo_criterion or args.dpo_criterion == 'enamine.avail' or 'freedom_space' in args.dpo_criterion:
        evaluator = None # for combined criterion, metrics have to be computed separately
        if args.metrics_detailed is None:
            raise ValueError('For combined/synthezisability criterion, detailed metrics file has to be provided')
        if not args.ignore_missing_scores:
            raise ValueError('For combined/synthezisability criterion, --ignore-missing-scores flag has to be set')
    else:
        raise ValueError(f"Unknown DPO criterion: {args.dpo_criterion}")
    
    # Make output directory
    dirname = f"dpo_{args.dpo_criterion.replace('.','_')}_{args.pocket}"
    if args.flex:
        dirname += '_flex'
    if args.normal_modes:
        dirname += '_nma'
    if args.toy:
        dirname += '_toy'
    processed_dir = Path(args.basedir, dirname)
    processed_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = processed_dir / 'chunks'
    chunks_dir.mkdir(exist_ok=True)

    if (chunks_dir / f'samples_{args.dpo_criterion}_{args.job_id}.csv').exists():
        logging.info(f"Samples already computed for criterion {args.dpo_criterion} job {args.job_id}, loading from file")
        samples = pd.read_csv(chunks_dir / f'samples_{args.dpo_criterion}_{args.job_id}.csv')
        samples = [dict(row) for _, row in samples.iterrows()]
        logging.info(f"Found {len(samples)} winning/losing samples")
    else:
        precomp_scores = None
        if args.metrics_detailed:
            logging.info(f'Loading precomputed scores from {args.metrics_detailed}')
            precomp_scores = pd.read_csv(args.metrics_detailed)
            precomp_scores = precomp_scores.drop_duplicates(subset=['sdf_file'])
            precomp_scores = precomp_scores.set_index('sdf_file')

        logging.info('Scanning sample directory...')
        samples_dir = Path(args.smplsdir)
        # scan dir
        sample_dirs = scan_smpl_dir(samples_dir, precomp_scores=precomp_scores, job_id=args.job_id, n_jobs=args.n_jobs)
        
        logging.info(f'Found {len(sample_dirs)} valid sample directories')
        
        if args.toy and args.n_jobs > 1:
            raise ValueError('Toy mode not supported with multiple jobs')
        if args.toy:
            args.toy_size = args.toy_size
            
        logging.info('Computing scores...')
        samples = compute_scores(
            sample_dirs, evaluator, args.dpo_criterion,
            n_pairs=args.n_pairs, toy=args.toy, toy_size=args.toy_size,
            precomp_scores=precomp_scores,
            ignore_missing_scores=args.ignore_missing_scores
        )
        logging.info(f'Found {len(samples)} winning/losing samples, saving to file')
        pd.DataFrame(samples).to_csv(Path(chunks_dir, f'samples_{args.dpo_criterion}_{args.job_id}.csv'), index=False)

    data_split = {}
    data_split['train'] = samples
    if args.toy:
        data_split['train'] = random.sample(samples, min(args.toy_size, len(data_split['train'])))

    failed = {}
    train_smiles = []

    for split in data_split.keys():

        logging.info(f"Processing {split} dataset...")

        ligands_w = defaultdict(list)
        ligands_l = defaultdict(list)
        pockets = defaultdict(list)

        tic = time()
        
        pbar = tqdm(data_split[split])
        for entry in pbar:
            pbar.set_description(f'#failed: {len(failed)}')
            res = process_single_entry(
                entry, args.pocket, args.flex, args.normal_modes
            )
            if res['status'] == 'failed':
                failed[(split, *res['key'])] = res['error']
                continue
            
            ligand_w = res['ligand_w']
            ligand_l = res['ligand_l']
            pocket = res['pocket']
            
            nerf_keys = ['fixed_coord', 'atom_mask', 'nerf_indices', 'length', 'theta', 'chi', 'ddihedral', 'chi_indices']
            for k in ['x', 'one_hot', 'bonds', 'bond_one_hot', 'v', 'nma_vec'] + nerf_keys + ['axis_angle']:
                if k in ligand_w:
                    ligands_w[k].append(ligand_w[k])
                    ligands_l[k].append(ligand_l[k])
                if k in pocket:
                    pockets[k].append(pocket[k])
            
            ligands_w['name'].append(res['ligand_w_name'])
            ligands_l['name'].append(res['ligand_l_name'])
            pockets['name'].append(res['pocket_name'])
            train_smiles.append(res['smiles_w'])
            train_smiles.append(res['smiles_l'])

        data = {'ligands_w': ligands_w, 
                'ligands_l': ligands_l,
                'pockets': pockets}
        torch.save(data, Path(chunks_dir, f'{split}_{args.job_id}.pt'))

        if split == 'train':
            np.save(Path(chunks_dir, f'train_smiles_{args.job_id}.npy'), train_smiles)

        print(f"Processing {split} set took {(time() - tic) / 60.0:.2f} minutes")

    # Write error report
    error_str = ""
    for k, v in failed.items():
        error_str += f"{'Split':<15}:  {k[0]}\n"
        error_str += f"{'Ligand W':<15}:  {k[1]}\n"
        error_str += f"{'Ligand L':<15}:  {k[2]}\n"
        error_str += f"{'Pocket':<15}:  {k[3]}\n"
        error_str += f"{'Error type':<15}:  {v[0]}\n"
        error_str += f"{'Error msg':<15}:  {v[1]}\n\n"

    with open(Path(chunks_dir, f'errors_{args.job_id}.txt'), 'w') as f:
        f.write(error_str)

    with open(Path(processed_dir, 'dataset_config.txt'), 'w') as f:
        f.write(str(args))

if __name__ == '__main__':
    main()