# HOLI-Lya
A repository containing the HOLI-Lya forest mocks code 

Set up a fresh environment at NERSC:
```
module load python
conda create -n holi-lya python=3.13
conda activate holi-lya
conda install numpy scipy numba h5py astropy healpy
```

We also need ```pyigm``` installed. Follow the installation instructions here: https://pyigm.readthedocs.io/en/latest/index.html

To run the full pipeline, either submit a Slurm job:
```
sbatch submit_lyamocks.sh
```
or from an interative node execute:
```
sh submit_lyamocks.sh
```