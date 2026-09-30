import os
import numpy as np

simstart = 1
simend = 9

#sims = np.arange(201,251)
#sims = np.array([241,242,243,244,245,246,247,248,249,250])

for ii in range(simstart,simend+1):
#for ii in sims:

    print('=================')
    print('Mock %d' %ii)
    os.system('mkdir skewers-%d' %ii)
    #os.system('mkdir /global/cfs/cdirs/desi/mocks/lya_forest/develop/cs-alpt/alpt_skewers/v5.0/skewers-%d' %ii)
    #os.system('mkdir /global/cfs/cdirs/desi/mocks/lya_forest/develop/cs-alpt/alpt_skewers/v5.0/skewers-%d/aux' %ii)

    os.system('cp *py skewers-%d' %ii)
    os.system('cp *DAT skewers-%d' %ii)
    os.system('cp *par skewers-%d' %ii)
    os.system('cp *sh skewers-%d' %ii)
    os.system('cp *fits skewers-%d' %ii)
    os.system('cp *webonx skewers-%d' %ii)
    os.system('cp -r webon skewers-%d' %ii)
    os.system('cp -r auxarr/*DAT skewers-%d' %ii)
    
    ff = open('skewers-%d/input_params.py' %ii, 'w')
    ff.write('nreal = %d\n' %ii)
    ff.write("version = 'v5.0'\n")
    ff.close()

    #os.system('cp extract_and_regrid_gaussian.py skewers-%d' %ii)
    #os.system('cp submit_lyamocks.sh skewers-%d' %ii)

    os.system('cd skewers-%d; sbatch submit_lyamocks.sh; cd ..' %ii)
    #os.system('cd skewers-%d; sh submit_lyamocks.sh; cd ..' %ii)
    #os.system('cd skewers-%d; python3 extract_and_regrid_gaussian.py; cd ..' %ii)
    print('=================')
