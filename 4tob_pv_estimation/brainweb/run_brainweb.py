# Estimate PVs for the Brainweb synthetic T1 images 
# The following methods will be run: 
# FAST (conventional volumetric)
# Toblerone (with FreeSurfer and FSL FIRST beforehand)
# RC method (cortex only)

# Scroll to the end for an overview of what happens 

# This script requires that FSL 6.0.1 and the HCP's 
# wb_command be installed 

import numpy as np 
import os.path as op 
import toblerone
import itertools 
import sys 
import multiprocessing
import nibabel 
import copy 
import subprocess
import scipy.io as sio
import glob

sys.path.append('..')
import image_scripts
from image_scripts import resample, do_RC, masked_vox_diff, restack
from toblerone.utils import STRUCTURES

ROOT = '/mnt/hgfs/Data/thesis_data/tob_pv_estimation/brainweb'
NUS = np.array([0,20,40])
NOISES = np.array([0,1,3,5,7,9])
VOXSIZES = np.arange(1,5)

def t1_name(nu, noise):
    return op.join(ROOT, 
        't1/t1_icbm_normal_1mm_pn%s_rf%s.nii.gz' % (noise, nu))

def anatdir(nu,noise):
    return op.join(ROOT, 'nu%s_noise%s.anat' % (nu,noise))
           
def refname(v):
    return op.join(ROOT, 'truth', 'ref%d.nii.gz' % v)

def truname(v):
    return op.join(ROOT, 'truth', 'tru%d.nii.gz' % v)

def fastdir():
    return op.join(ROOT, 'fast')

def tobdir(nu,noise,v):
    return op.join(ROOT, 'tob_nu%s_noise%s_%s' % (nu,noise,v))

def tobname(nu,noise,v):
    td = tobdir(nu,noise,v)
    return op.join(td, 'nu%s_noise%s_all_stacked.nii.gz' % (nu,noise))

def fastname(nu,noise,v):
    return op.join(ROOT, 'fast', 'fast_nu%d_noise%d_%d.nii.gz' % (nu,noise,v))

def make_anat_dir_mp(nu,noise):

    suff = 'nu%s_noise%s' % (nu,noise)
    img = op.join(ROOT, suff + '.nii.gz')
    od = op.join(ROOT, suff + '.anat')
    t1 = t1_name(nu,noise)

    if not toblerone.utils.check_anat_dir(od):
        toblerone.fsl_fs_anat(struct=t1, out=od)

def make_refs(): 

    ref1 = op.join(ROOT, 't1', 't1_icbm_normal_1mm_pn0_rf0.nii.gz')
    ref1 = nibabel.load(ref1)
    ref_aff = ref1.affine
    FoV = np.array([181,217,181])

    for v in range(1,5):

        outname = op.join(ROOT, 'truth', 'ref%d.nii.gz' % v)
        dims = np.ceil(FoV/v).astype(np.int16)
        new_aff = copy.copy(ref_aff)
        new_aff[range(3), range(3)] *= v 
        new_orig = ref_aff[0:3,3] - (0.5 * ref_aff[range(3), range(3)]) + (0.5 * new_aff[range(3), range(3)])
        new_aff[0:3,3] = new_orig
        cmd = 'wb_command -volume-create %d %d %d %s -sform ' % (*dims, outname)
        for x in new_aff[0:3,:].flatten(): 
            cmd += ('%1.2f ' % x)
        subprocess.run(cmd, shell=True)


def make_tobs(): 

    nus_noises = itertools.product(NUS, NOISES)

    for (nu,noise) in nus_noises:
        for v in VOXSIZES: 
            td = tobdir(nu,noise,v)
            toblerone.utils._weak_mkdir(td)
            tobfile = tobname(nu, noise, v)
            ref = refname(v)
            spc = toblerone.classes.ImageSpace(ref)

            if not op.isfile(tobfile):
                try:                                
                    stacked = restack(td, '')
                    spc.save_image(stacked, tobfile)

                except Exception as e: 
                    anat = anatdir(nu,noise)
                    pvs, _ = toblerone.estimate_all(anat=anat, 
                        ref=ref, struct2ref='I')
                    for key,img in pvs.items():
                        outpath = op.join(td, 'nu%s_noise%s_%s.nii.gz' % (nu,noise,key))
                        spc.save_image(img, outpath) 


def make_fasts(): 

    fastdir = op.join(ROOT, 'fast')
    toblerone.utils._weak_mkdir(fastdir)

    nus_noises = itertools.product(NUS, NOISES)
    for (nu,noise) in nus_noises:
        for v in VOXSIZES: 

            outname = op.join(fastdir, 'fast_nu%s_noise%s_%s.nii.gz' % (nu,noise,v))
            if not op.isfile(outname):
                if (v == 1): 

                    tmpout = op.join(fastdir, 'fast_nu%s_noise%s_%s_noref.nii.gz' % (nu,noise,v))
                    anat = anatdir(nu,noise)
                    csffile,gmfile,wmfile = map(lambda t: op.join(
                        ROOT, anat, 'T1_fast_pve_%d.nii.gz' % t), range(3)) 

                    # Merge the single channel images into one (in -t dim)
                    csffile = op.join(ROOT, anat, 'T1_fast_pve_csf.nii.gz')
                    gm = nibabel.load(gmfile).get_fdata()
                    wm = nibabel.load(wmfile).get_fdata()
                    csf = 1.0 - (gm + wm)
                    toblerone.classes.ImageSpace.save_like(gmfile, csf, csffile)
                    cmd = 'fslmerge -t %s %s %s %s' % (tmpout, gmfile, wmfile, csffile)
                    subprocess.run(cmd, shell=True)

                    ref = refname(v)
                    resample(tmpout, outname, ref)
                    subprocess.run('rm %s' % tmpout, shell=True)

                else:
                    ref = refname(v)
                    base = op.join(fastdir, 'fast_nu%s_noise%s_1.nii.gz' % (nu,noise))
                    resample(base, outname, ref)

def RCdir(nu,noise):
    return op.join(ROOT, 'RC_nu%d_noise%d' % (nu,noise))

def RCname(nu,noise,v):
    rcdir = RCdir(nu,noise)
    return op.join(rcdir, 'RC_nu%s_noise%s_%s.nii.gz' % (nu,noise,v))

def make_RCs():

    nus_noises = itertools.product(NUS, NOISES)
    for (nu,noise) in nus_noises:

        rcdir = RCdir(nu,noise)
        toblerone.utils._weak_mkdir(rcdir)
        surfdir = op.join(anatdir(nu,noise), 'fs/surf')

        for side,surf in itertools.product(['lh', 'rh'], ['white', 'pial']):
            ins = op.join(surfdir, '%s.%s' % (side, surf))
            outs = op.join(surfdir, '%s.%s.surf.gii' % (side,surf))
            if not op.isfile(outs):
                cmd = 'mris_convert --to-scanner %s %s' % (ins, outs)
                subprocess.run(cmd, shell=True)

        RWS = op.join(surfdir, 'rh.white.surf.gii')
        RPS = op.join(surfdir, 'rh.pial.surf.gii')
        LMS = op.join(surfdir, 'lh.mid.surf.gii')
        LPS = op.join(surfdir, 'lh.pial.surf.gii')
        RMS = op.join(surfdir, 'rh.mid.surf.gii')
        LWS = op.join(surfdir, 'lh.white.surf.gii')

        for v in VOXSIZES: 
            outname = RCname(nu,noise,v)
            if not op.isfile(outname):
                do_RC(RWS=RWS,LWS=LWS,LPS=LPS,LMS=LMS,RPS=RPS,RMS=RMS,
                    outname=outname, vox=v, ref=refname(v))


def sum_to_ref(img, ref): 

    input = nibabel.load(img)
    refSpc = toblerone.classes.ImageSpace(ref)
    inSpc = toblerone.classes.ImageSpace(img)

    factor = refSpc.vox_size / inSpc.vox_size
    if np.any(np.mod(factor, np.ones_like(factor))): 
        raise RuntimeError("Ratio of reference to input voxel size must be an integer")

    if np.any(factor != factor[0]):
        raise RuntimeError("Rescaling factor must be constant in all dimensions")

    inshp = input.shape 

    # Create zeros, sized according to the FoV of reference with voxels sized
    # according to the reference. Then write in the input data 
    outshp = tuple( (d * refSpc.vox_size[idx] / inSpc.vox_size[idx]).astype(np.int16) for idx,d in enumerate(refSpc.size) )
    data = input.get_fdata()

    if len(data.shape) == 3: 
        frames = 1 
    elif len(data.shape) == 4: 
        frames = data.shape[3]
    else: 
        raise RuntimeError("Data size greater than 4D")

    output = np.zeros((*outshp, frames), dtype=np.float32)
    output[:inshp[0], :inshp[1], :inshp[2], :] = data[:,:,:,...]

    # Now sum across blocks for output, and don't forget to average out!
    summedout = toblerone.resampling._sumArrayBlocks(output, (*factor, 1)) / (factor[0] ** 3)

    return np.squeeze(summedout)


def safeload(path, ref):
    img = nibabel.load(path).get_fdata()
    refspc = toblerone.classes.ImageSpace(ref)
    if not np.all(img.shape[0:3] == refspc.size):
        raise RuntimeError("Image does not match reference")
    return img.reshape(-1,3)


def automask(img):
    return np.logical_or(img[:,0] > 0, img[:,1] > 0)


def main():
  
    # There are 18 combinations of NU and Noise level
    nus_noises = itertools.product(NUS, NOISES)

    # Prepare an FSL_anat dir (w/ FS output added) for each nu,noise pair 
    with multiprocessing.Pool() as p:
        p.starmap(make_anat_dir_mp, nus_noises)

    # All methods require the reference spaces to be generated first 
    make_refs()      

    # Then toblerone, FAST and RC, in that order 
    make_tobs()
    make_fasts()
    make_RCs()
    
    # Analysis below 
    meths = ['fast', 'tob', 'rc']

    # Output arrays
    # Tissue is always [GM, WM]
    # Total tissue volumes, dims: NU x noise x voxel size x method x tissue 
    sums = np.zeros((len(NUS), len(NOISES), len(VOXSIZES), len(meths), 2), dtype=np.float32)

    # Voxel-wise differences, sized the same as above 
    voxs = np.zeros_like(sums)

    # Individual structure volumes, dims: NU x noise x voxel size x tissue x structure
    structs = np.zeros((len(NUS), len(NOISES), len(VOXSIZES), 2, len(STRUCTURES)+1), dtype=np.float32)

    for vidx,v in enumerate(VOXSIZES): 

        # For each voxel size, we load the reference space and then each method's 1mm 
        # estimate with NU 0 and noise 0. We then calculate each method's reference at
        # other voxel sizes by summing across perfect integer multiples of the 1mm 
        # results (which ensures no blurring of voxels). The safeload() and sum_to_ref()
        # methods handle this 
        ref = refname(v)
        tob_ref = sum_to_ref(tobname(0,0,1), ref)
        fast_ref = sum_to_ref(op.join(ROOT, 'fast', 'fast_nu0_noise0_%s.nii.gz' % 1), ref)
        rc_ref = sum_to_ref(RCname(0,0,1), ref)

        # Optional block: save each method's reference at each voxel size 
        refspc = toblerone.classes.ImageSpace(ref)
        refspc.save_image(tob_ref, op.join(ROOT, 'tob_ref_%d.nii.gz' % v))
        refspc.save_image(fast_ref, op.join(ROOT, 'fast_ref_%d.nii.gz' % v))
        refspc.save_image(rc_ref, op.join(ROOT, 'rc_ref_%d.nii.gz' % v))

        tob_ref, fast_ref, rc_ref = [ x.reshape(-1,3) for x in [tob_ref, fast_ref, rc_ref] ] 

        for (nuidx,nu),(noidx,noise) in itertools.product(enumerate(NUS), enumerate(NOISES)): 
            
            fast = safeload(op.join(ROOT, 'fast', 'fast_nu%s_noise%s_%s.nii.gz' % (nu,noise,v)), ref)
            tob = safeload(tobname(nu,noise,v), ref)
            rc = safeload(RCname(nu,noise,v), ref)

            # Total tissue volume 
            sums[nuidx,noidx,vidx,0,:] = tob[:,0:2].sum(0)
            sums[nuidx,noidx,vidx,1,:] = fast[:,0:2].sum(0)
            sums[nuidx,noidx,vidx,2,:] = rc[:,0:2].sum(0)

            # Voxel-wise differences
            voxs[nuidx,noidx,vidx,0,:] = masked_vox_diff(tob_ref, tob)
            voxs[nuidx,noidx,vidx,1,:] = masked_vox_diff(fast_ref, fast)
            voxs[nuidx,noidx,vidx,2,:] = masked_vox_diff(rc_ref, rc)

            # Tissue volume of each subcortical structure (Toblerone only)
            for sidx,struct in enumerate(STRUCTURES):

                tob_str = nibabel.load(op.join(tobdir(nu,noise,v), 
                    'nu%d_noise%d_%s.nii.gz' % (nu,noise,struct))).get_fdata()
                structs[nuidx,noidx,vidx,0,sidx] = tob_str.sum()

            # Add in the cortex (Toblerone and RC as well)
            tob_str = nibabel.load(op.join(tobdir(nu,noise,v), 
                'nu%d_noise%d_cortex_GM.nii.gz' % (nu,noise))).get_fdata()
            structs[nuidx,noidx,vidx,0,-1] = tob_str.sum()
            structs[nuidx,noidx,vidx,1,-1] = rc[:,0].sum()


    sio.savemat('brainweb_data.mat', {
        'sums': sums,
        'structs': structs, 
        'voxs': voxs, 
        })


if __name__ == "__main__":
    
    # Take a look at the main() function above ^
    main()
