import argparse
import warnings

from Bio.PDB import PDBParser
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import DataLoader
from functools import partial

warnings.filterwarnings("ignore")

from src import utils
from src.data.dataset import ProcessedLigandPocketDataset
from src.data.data_utils import TensorDict, process_raw_pair, prepare_ligand
from src.model.lightning import DrugFlow
from src.constants import atom_encoder, bond_encoder

from tqdm import tqdm
from pdb import set_trace


def aggregate_metrics(table):
    agg_col = 'posebusters'
    total = 0
    table[agg_col] = 0
    for column in table.columns:
        if column.startswith(agg_col) and column != agg_col:
            table[agg_col] += table[column].fillna(0).astype(float)
            total += 1
    table[agg_col] = table[agg_col] / total

    agg_col = 'reos'
    total = 0
    table[agg_col] = 0
    for column in table.columns:
        if column.startswith(agg_col) and column != agg_col:
            table[agg_col] += table[column].fillna(0).astype(float)
            total += 1
    table[agg_col] = table[agg_col] / total

    agg_col = 'chembl_ring_systems'
    total = 0
    table[agg_col] = 0
    for column in table.columns:
        if column.startswith(agg_col) and column != agg_col and not column.endswith('smi'):
            table[agg_col] += table[column].fillna(0).astype(float)
            total += 1
    table[agg_col] = table[agg_col] / total
    return table


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--protein', type=str, required=True, help="Input PDB file.")
    p.add_argument('--ligand', type=str, required=True, help="SMILES string of the ligand to dock")
    p.add_argument('--reference', type=str, required=True, help="SDF file with reference ligand used to define the pocket.")
    p.add_argument('--checkpoint', type=str, required=False, default='models/drugdock_e96.ckpt', help="Model checkpoint file.")
    p.add_argument('--output', type=str, required=False, default='results.sdf', help="Output file.")
    p.add_argument('--n_samples', type=int, required=False, default=10, help="Number of sampled molecules.")
    p.add_argument('--batch_size', type=int, required=False, default=32, help="Batch size.")
    p.add_argument('--pocket_distance_cutoff', type=float, required=False, default=8.0, help="Distance cutoff to define the pocket around the reference ligand.")
    p.add_argument('--n_steps', type=int, required=False, default=None, help="Number of denoising steps.")
    p.add_argument('--device', type=str, required=False, default='cuda:0', help="Device to use.")
    p.add_argument('--seed', type=int, required=False, default=None, help="Random seed.")
    args = p.parse_args()

    if args.seed is not None:
        utils.set_deterministic(seed=args.seed)

    utils.disable_rdkit_logging()
    
    # Loading model
    chkpt_path = Path(args.checkpoint)
    chkpt_name = chkpt_path.parts[-1].split('.')[0]
    model = DrugFlow.load_from_checkpoint(args.checkpoint, map_location=args.device, strict=False)
    model.setup(stage='generation')
    model.batch_size = model.eval_batch_size = args.batch_size
    model.eval().to(args.device)
    if args.n_steps is not None:
        model.T = args.n_steps

    # Preparing input
    pdb_model = PDBParser(QUIET=True).get_structure('', args.protein)[0]
    rdmol = Chem.SDMolSupplier(str(args.reference))[0]
    _, pocket = process_raw_pair(
        pdb_model, rdmol,
        dist_cutoff=args.pocket_distance_cutoff,
        pocket_representation=model.pocket_representation,
        compute_nerf_params=True,
        nma_input=args.protein if model.dynamics.add_nma_feat else None
    
    )

    rdmol = Chem.MolFromSmiles(args.ligand)
    assert rdmol is not None, 'Input ligand is invalid'
    AllChem.Compute2DCoords(rdmol)
    ligand = prepare_ligand(rdmol, atom_encoder, bond_encoder)
    ligand['name'] = 'ligand'

    dataset = [{'ligand': ligand, 'pocket': pocket} for _ in range(args.batch_size)]
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size, 
        collate_fn=partial(ProcessedLigandPocketDataset.collate_fn, ligand_transform=None),
        pin_memory=True
    )

    sampled_molecules = []
    Path(args.output).parent.absolute().mkdir(parents=True, exist_ok=True)
    print(f'Will generate {args.n_samples} samples')

    with tqdm(total=args.n_samples) as pbar:
        while len(sampled_molecules) < args.n_samples:
            for i, data in enumerate(dataloader):
                new_data = {
                    'ligand': TensorDict(**data['ligand']).to(args.device),
                    'pocket': TensorDict(**data['pocket']).to(args.device),
                }
                rdmols, rdpockets, _ = model.sample(
                    new_data,
                    n_samples=1,
                    timesteps=args.n_steps,
                    num_nodes='ground_truth',
                )
                sampled_molecules.extend(rdmols)
                pbar.update(len(rdmols))

    # Write results
    utils.write_sdf_file(args.output, sampled_molecules)
