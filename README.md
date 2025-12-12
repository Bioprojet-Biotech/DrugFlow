# DrugFlow-Dock

## Setup

### Conda Environment

Create a conda/mamba environment 
```bash
conda env create -f environment.yaml -n drugflow
conda activate drugflow
```

and add the Gnina executable for docking score computation
```bash
wget https://github.com/gnina/gnina/releases/download/v1.1/gnina -O $CONDA_PREFIX/bin/gnina
chmod +x $CONDA_PREFIX/bin/gnina
```

### Docker Container

A pre-built Docker container is available on [DockerHub](https://hub.docker.com/r/igashov/drugflow):

```bash
docker pull igashov/drugflow:0.0.3
```

## Usage

To dock a molecule, run the following command:

```bash
python dock.py \
       --protein examples/kras.pdb \
       --ligand "C[CH]1CN(C)CCN1c2cccc(n2)c3noc(n3)[C]4(C)CCCc5sc(N)c(C#N)c45" \
       --reference examples/kras_ref_ligand.sdf
```

Docking conformations of the molecule will be saved in SDF-file `results.sdf`.  For more options, see:
```bash
python dock.py --help
```
