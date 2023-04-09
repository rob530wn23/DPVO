import os
import glob
from joblib import Parallel, delayed

os.chdir('datasets/TartanAir')
xs = glob.glob('*')
# for x in xs:
#     os.system("rm -rf " + x + '/' + x)
#
# exit()
all_zips = glob.glob('*/*/*.zip')

def unzip(zip):
    os.system('unzip ' + zip)

Parallel(n_jobs=8)(delayed(unzip)(zip) for zip in all_zips)

