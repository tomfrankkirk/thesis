# Produce PVs for the simulated cortical surfaces via following methods: 
# Ground truth 
# Toblerone + (resampling variant)
# Ribbon-constrained method (+ resampling variant)
# NeuropolyPVE (+ resampling variant)

# Finally this script will analyse the results and store the 
# output as 'sim_surface_data.mat'

# This script requires that the HCP's wb_command be installed 

import sys 
import os.path as op
import numpy as np
import nibabel
import scipy.io
import toblerone
import subprocess

sys.path.append('..')
import image_scripts
from image_scripts import masked_vox_diff

VOXSIZES = np.arange(1, 3.2, 0.2)
ROOT = '/mnt/hgfs/Data/thesis_data/tob_pv_estimation/sim_surfaces/surf'

def refname(v):
    return op.join(ROOT, 'ref_{:1.2f}.nii'.format(v))

def truname(v):
    return op.join(ROOT, 'tru_{:1.2f}.nii'.format(v))

def make_refs():
    """Produce the reference spaces within which all methods will estimate PVs"""

    orig = np.array([-75, -75, -75])
    FoV = np.array([150, 150, 150])
    for v in VOXSIZES:
        ref = refname(v)
        if not op.isfile(ref):
            image_scripts.make_reference(orig, FoV, v, ref)

def summer(a, v):
    return np.sum(a[:,0:2], axis=0) * (v ** 3)
            

def main(root):
    
    ROOT = root

    # All methods require the reference spaces to be created first
    make_refs()

    # Call the MATLAB script to calculate ground truth, RC and Neuropoly results 
    # (this last method is sloooooooooow)
    matpath = "/opt/Matlab/R2017a/bin/matlab"
    if not op.isfile(matpath):
        raise RuntimeError("Please edit this script to contain your matlab path")
    cmd = '%s -nosplash -nodesktop -r \'ROOT = \"%s/\"; cd sim_surfaces; matlab_scripts; exit\'' % (matpath, ROOT)
    if subprocess.run(cmd, shell=True).returncode != 0:
        raise RuntimeError("Error during matlab execution")

    # Estimate using Toblerone 
    LPS = op.join(ROOT, 'lh.pial')
    LWS = op.join(ROOT, 'lh.white')

    for vidx,v in enumerate(VOXSIZES):
        outname = op.join(ROOT, 'tob_{:1.2f}.nii'.format(v))
        ref = refname(v)

        if not op.exists(outname):
            refSpace = toblerone.classes.ImageSpace(ref)
            out, _, _= toblerone.estimate_cortex(LWS=LWS, LPS=LPS, ref=ref, struct2ref='I', cores=4)
            refSpace.saveImage(out, outname)

        base = op.join(ROOT, 'tru_{:1.2f}.nii'.format(v))
        for v2 in VOXSIZES[vidx+1:]:
            ref2 = refname(v2)
            out2name = op.join(ROOT, 'tru_{:1.2f}_resamp_{:1.2f}.nii.gz'.format(v, v2))
            if not op.isfile(out2name):
                image_scripts.resample(base, out2name, ref2)


    # And resampling for the RC2 and Neuro2 methods: 
    rcbase = op.join(ROOT, 'rc_1.00.nii')
    neurobase = op.join(ROOT, 'neuro1_1.00.nii')
    for v in VOXSIZES[1:]:
        ref = refname(v)
        rc2 = op.join(ROOT, 'rc_2_%1.2f.nii.gz' % v)
        neuro2 = op.join(ROOT, 'neuro_2_%1.2f.nii.gz' % v)
        if not op.isfile(rc2):
            image_scripts.resample(rcbase, rc2, ref)
        if not op.isfile(neuro2):
            image_scripts.resample(neurobase, neuro2, ref)


    # Analysis below 
    loader = lambda path: nibabel.load(path).get_fdata().reshape(-1,3)

    # Output arrays and their dimensions (tissues is always [GM, WM]): 
    # Voxel-wise errors, dims: voxels x methods x tissues 
    voxs = np.zeros((VOXSIZES.size, 5, 2))

    # Total tissue volumes, dims: voxels x methods x tissues 
    sums = np.zeros((VOXSIZES.size, 6, 2))

    # Voxel-wise error between native and resampled Toblerone, dims: voxels x tissues
    resamp = np.zeros((VOXSIZES.size, VOXSIZES.size, 2))

    for vidx, v in enumerate(VOXSIZES): 

        tob = loader(op.join(ROOT, 'tob_%1.2f.nii' % v))
        tru = loader(op.join(ROOT, 'tru_%1.2f.nii' % v))
        rc = loader(op.join(ROOT, 'rc_%1.2f.nii' % v))
        neuro = loader(op.join(ROOT, 'neuro1_%1.2f.nii' % v))
        if vidx > 0: 
            rc2 = loader(op.join(ROOT, 'rc_2_%1.2f.nii.gz' % v))
            neuro2 = loader(op.join(ROOT, 'neuro_2_%1.2f.nii.gz' % v))

        # Total tissue volumes 
        sums[vidx,5,:] = summer(tru,v)
        sums[vidx,0,:] = summer(tob,v)
        sums[vidx,1,:] = summer(rc,v)
        sums[vidx,3,:] = summer(neuro,v)
        if vidx > 0: 
            sums[vidx,2,:] = summer(rc2,v)
            sums[vidx,4,:] = summer(neuro2,v)

        # Filter to just voxels containing (GM & WM) or (GM & CSF)
        # ie, excluding those wholly within the cortex
        fltr = np.logical_or(
            ((tru[:,0] > 0) & (tru[:,1] > 0)), 
            ((tru[:,0] > 0) & (tru[:,2] > 0))
        )

        # Mean of abs voxel-wise differences in each tissue 
        voxs[vidx, 0] = masked_vox_diff(tru, tob, fltr)
        voxs[vidx, 1] = masked_vox_diff(tru, rc, fltr)
        voxs[vidx, 3] = masked_vox_diff(tru, neuro, fltr)
        if vidx > 0: 
            voxs[vidx, 2] = masked_vox_diff(tru, rc2, fltr)
            voxs[vidx, 4] = masked_vox_diff(tru, neuro2, fltr)


        # Resampling methods  
        for v2idx, v2 in enumerate(VOXSIZES[:vidx]):
            tru2 = loader(op.join(
                ROOT, 'tru_{:1.2f}_resamp_{:1.2f}.nii.gz'.format(v2, v)))
            resamp[v2idx,vidx,:] = masked_vox_diff(tru, tru2, fltr)

    scipy.io.savemat('sim_surface_data.mat', {
        'resamp': resamp, 'voxs': voxs, 'sums': sums
        })


if __name__ == "__main__":

    # Take a look at the main() function above ^ 
    main(ROOT)