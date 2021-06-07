# Script to run all Toblerone evaluation and analysis 

import sys
sys.path.append('./brainweb')
sys.path.append('./HCP_retest')
sys.path.append('./sim_surfaces')

from brainweb import run_brainweb
from HCP_retest import run_HCP
from sim_surfaces import run_sim_surfaces

# Path to directory in which all the raw data is stored 
ROOT = '/mnt/hgfs/Data/toblerone_evaluation_data'

if __name__ == "__main__":
    
    # The simulated surfaces will run within the 'sim_surfaces' subdir 
    print("Simulated surfaces")
    run_sim_surfaces.main(ROOT + '/sim_surfaces/surf')

    # and 'brainweb' for the brainweb data
    print("Brainweb")
    run_brainweb.main(ROOT + '/brainweb')

    # and finally 'HCP_retest', which must contain the subdirs:
    # 'test': the 45 subject directories from the structural preprocessed release, eg 103818
    # 'retest': the retest data for the same 45 subjects 
    print("HCP test/retest")
    run_HCP.main(ROOT + '/HCP_retest')
    

# Finally, in order to generate the graphs used in the paper, look in each subdirectory and run 
# the appropriate _analysis script (eg brainweb_analysis). 
