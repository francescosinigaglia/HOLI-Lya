#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=128 # tasks out of 128  
#SBATCH -C cpu             
#SBATCH -q regular               
#SBATCH -J job_name
#SBATCH -o log_filename            
#SBATCH --mail-type=ALL   
#SBATCH -t 03:00:00                                                                                                       
#SBATCH --mem=475G 

module load python
export NUMBA_NUM_THREADS=128

# QSO catalog
python3 make_catalog_lightcone.py
python3 write_catalog.py
python3 rewrite_qso_cat_for_xcf.py

# DLA catalog
conda run -n pyigm python3 make_catalog_DLA.py
python3 write_catalog_DLA.py

# Lya skewers
python3 extract_and_regrid_gaussian.py
