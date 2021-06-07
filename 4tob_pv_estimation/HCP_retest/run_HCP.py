# HCP retest

import os
import numpy as np 
import os.path as op 
import subprocess
import toblerone as t 
from toblerone.utils import STRUCTURES
import functools
import glob 
import multiprocessing
import nibabel
import scipy.io as sio

from image_scripts import do_RC, resample, masked_vox_diff, restack

T1ROOT = 'T1w_acpc_dc_restore_brain'
VOXSIZES = np.hstack(([0.7], np.arange(1.0, 4.2, 0.4)))
EXCLUDE = []

ROOT = '/mnt/hgfs/Data/thesis_data/tob_pv_estimation/HCP_retest'

def SUBIDS():
    dirs = glob.glob(op.join(ROOT, 'test', '*'))
    subdirs = [ d for d in dirs if (op.isdir(d) and ((op.split(d)[1] != 'refs') 
        and (op.splitext(d)[1] != '.zip') and (op.split(d)[1] != 'fast'))) ]

    subids = [ op.split(d)[1] for d in subdirs ] 
    subids = [ s for s in subids if (s not in EXCLUDE) ]
    return subids

def refname(v):
    return op.join(ROOT, 'refs', 'ref_%1.1f.nii.gz' % v)

def subdir(ID):
    return op.join(ID, 'T1w')

def outdir(ID):
    o = op.join(subdir(ID), 'processed')
    t.utils._weak_mkdir(o)
    return o 

def outname_formethod(outdir, method, vs):
    return op.join(outdir, '%s_%1.1f.nii.gz' % (method, vs))

def map_over_subjects(run, nSubs, cores, func):
    subids = SUBIDS()
    subdirs = [ op.join(ROOT, run, d) for d in subids ]
    outdirs = [ op.join(d, 'T1w', 'processed') 
        for (d,ID) in zip(subdirs, subids) ]

    cap = min([nSubs, len(subdirs)])
    subdirs = subdirs[0:cap]

    for d in outdirs:
        if not op.isdir(d):
            os.mkdir(d)

    if cores == 1:
        for sub in subdirs:
            func(run, sub)
            
    else: 
        f = functools.partial(func, run)
        with multiprocessing.Pool(cores) as p:
            p.map(f, subdirs)

def shell(cmd):
    subprocess.run(cmd, shell=True)

def references():

    t.utils._weak_mkdir(op.join(ROOT, 'refs'))
    orig = np.array([90, -126, -72])
    FoV = np.array([260,311,260]) * 0.7 

    for v in VOXSIZES: 
        ref = refname(v)

        if not op.isfile(ref):
            dims = np.ceil(FoV / v)
            cmd = 'wb_command -volume-create %d %d %d ' % (dims[0], dims[1], dims[2])
            cmd += '%s -sform -%.3f 0 0 %.3f '  % (ref, v, orig[0])
            cmd += '0 %.3f 0 %.3f ' % (v, orig[1])
            cmd += '0 0 %.3f %.3f ' % (v, orig[2])

            shell(cmd)



def fast_subject(run, ID):

    sd = subdir(ID)
    od = outdir(ID)

    base_brain = op.join(od, 'fast_base_brain.nii.gz')
    base_cortex = op.join(od, 'fast_base_cortex.nii.gz')

    if not op.isfile(base_brain):

        os.chdir(sd)
        base = op.join(od, 'fast_base.nii.gz')

        if not op.isfile(base):
            cmd = 'fast -N %s' % (T1ROOT + '.nii.gz')
            shell(cmd)

        # Merge the single channel images into one (in -t dim)
        csffile = T1ROOT + "_pve_csf.nii.gz"
        f1 = T1ROOT + "_pve_1" + ".nii.gz"
        f2 = T1ROOT + "_pve_2" + ".nii.gz"
        gm = nibabel.load(f1).get_fdata()
        wm = nibabel.load(f2).get_fdata()
        csf = 1.0 - (gm + wm)
        t.classes.ImageSpace.save_like(f1, csf, csffile)
        cmd = 'fslmerge -t %s %s %s %s' % (base, f1, f2, csffile)
        shell(cmd)

        # Mask using the ribbon 
        base_brain = op.join(od, 'fast_base_brain.nii.gz')
        brainmaskf = op.join(sd, 'ribbon.nii.gz')
        brainmask = nibabel.load(brainmaskf).get_fdata().round().astype(np.int32)
        ctxmask = np.logical_or(brainmask == 3, brainmask == 42)
        ctxmaskf = op.join(od, 'cortex_mask.nii.gz')
        t.classes.ImageSpace.save_like(base, ctxmask, ctxmaskf)
        cmd = 'fslmaths %s -mas %s %s' % (base, ctxmaskf, base_cortex)
        shell(cmd)
        cmd = 'fslmaths %s -mas %s %s' % (base, brainmaskf, base_brain)
        shell(cmd)

    for v in VOXSIZES:

        ref = refname(v)

        for base, meth in zip([base_brain], ['fast']):
            outname = outname_formethod(od, meth, v)

            if (not op.isfile(outname)) and (v != 0.7):
                resample(base, outname, ref)



def toblerone_subject(run, ID):

    id_n = op.split(ID)[1]

    if run == 'test':
        sd = op.join(ROOT, run, id_n, 'T1w')

    else: 
        sd = op.join(ROOT, 'retest', id_n, 'T1w')

    t1 = op.join(sd, T1ROOT + '.nii.gz')
    RWS = op.join(sd, 'Native', '%s.R.white.native.surf.gii' % id_n) 
    RPS = op.join(sd, 'Native', '%s.R.pial.native.surf.gii' % id_n) 
    LWS = op.join(sd, 'Native', '%s.L.white.native.surf.gii' % id_n) 
    LPS = op.join(sd, 'Native', '%s.L.pial.native.surf.gii' % id_n) 
    first = op.join(sd, 'processed', 'first')

    od = outdir(ID)
    t.utils._weak_mkdir(od)

    for v in VOXSIZES: 

        tobfile = od + '/tob_all_stacked_%1.1f.nii.gz' % v
        ref = refname(v)
        spc = t.classes.ImageSpace(ref)

        if not op.isfile(tobfile):
            try: 
                stacked = restack(od, '_%1.1f' % v)
                spc.save_image(stacked, tobfile)

            except Exception as e: 
                pvs, _ = t.estimate_all(LWS=LWS, LPS=LPS, RWS=RWS, RPS=RPS, 
                    ref=ref, struct2ref='I', fastdir=sd, firstdir=first, struct=t1, cores=8)
                for key,img in pvs.items():
                    outpath = outname_formethod(od, 'tob_' + key, v)
                    spc.save_image(img, outpath) 


def first_subject(run, ID):

    od = outdir(ID)
    fd = op.join(od, 'first')
    t1 = op.join(subdir(ID), 'T1w_acpc_dc_restore.nii.gz')
    if not op.isfile(op.join(fd, 'T1w_acpc_dc_restore' + '-BrStem_first.vtk')):
        t.utils._runFIRST(t1, fd)

def loader(run, sub, method, vox, silent):

    s = op.join(ROOT, run, sub, 'T1w/processed')

    if method == 'tob':
        fname = 'tob_all_stacked_%1.1f' % vox
    elif method == 'fast':
        if (vox == 0.7):
            fname = 'fast_base_brain'
        else: 
            fname = 'fast_%1.1f' % vox
    else:
        fname = '%s_%1.1f' % (method, vox)

    s = op.join(s, '%s.nii.gz' % (fname))

    try: 
        return nibabel.load(s).get_fdata().reshape(-1,3)
    except Exception as e: 
        if silent: 
            refsize = nibabel.load(refname(vox)).header['dim'][1:4]
            return np.zeros((np.prod(refsize), 3))
        else:
            raise e 


def RC_subject(run, ID): 

    id_n = op.split(ID)[1]

    od = outdir(ID)
    surfdir = op.join(subdir(ID), 'Native')
    space = 'native'

    RWS = op.join(surfdir, '%s.R.white.%s.surf.gii' % (id_n,space)) 
    RPS = op.join(surfdir, '%s.R.pial.%s.surf.gii' % (id_n,space)) 
    LWS = op.join(surfdir, '%s.L.white.%s.surf.gii' % (id_n,space)) 
    LPS = op.join(surfdir, '%s.L.pial.%s.surf.gii' % (id_n,space)) 
    RMS = op.join(surfdir, '%s.R.mid.%s.surf.gii' % (id_n,space)) 
    LMS = op.join(surfdir, '%s.L.mid.%s.surf.gii' % (id_n,space)) 

    for v in VOXSIZES:
        outname = outname_formethod(od, 'RC', v)

        if not op.isfile(outname):
            ref = refname(v)
            PVs = do_RC(LPS=LPS, RPS=RPS, LWS=LWS, RWS=RWS, RMS=RMS, LMS=LMS, outname=outname, vox=v, ref=ref)


def main(root): 

    root = ROOT

    print("Producing references")
    references()

    cores = 8
    nSubs = 45

    if True: 
    
        runs = ['test', 'retest']

        for run in runs: 

            print("First", run)
            map_over_subjects(run, nSubs, cores, first_subject)
                
            print("RC", run)
            map_over_subjects(run, nSubs, 1, RC_subject)

            print("FAST", run)
            map_over_subjects(run, nSubs, cores, fast_subject)

            print("Toblerone", run)
            map_over_subjects(run, nSubs, 3, toblerone_subject)

    if True: 
        print("Analysis")

        # Matrix is sized: subs x vox x methods x runs x tissues
        methods = ['tob', 'fast', 'rc']
        subs = SUBIDS()[0:nSubs]
        sums = np.zeros((nSubs, len(VOXSIZES), len(methods), 2, 2), dtype=np.float32)
        voxs = np.zeros((nSubs, len(VOXSIZES), len(methods), 2, 2), dtype=np.float32)
        hist_bins = np.arange(0, 1.05, 0.05)
        diffs = np.zeros((nSubs, len(VOXSIZES), 2, 2, hist_bins.size -1), dtype=np.float32)
        all_structs = STRUCTURES + ['cortex_GM']
        structs = np.zeros((nSubs, 2, len(all_structs)), dtype=np.float32)

        summer = lambda a: np.sum(a[:,0:2], axis=0)
    
        for (sidx,sub) in enumerate(subs):        
            for (vidx,vox) in enumerate(VOXSIZES): 
                silent = False 
                tob, fast, rc = [ loader('test', sub, m, vox, silent) for m in methods ]
                tob2, fast2, rc2 = [ loader('retest', sub, m, vox, silent) for m in methods ]

                sums[sidx,vidx,0,0,:] = summer(tob)
                sums[sidx,vidx,1,0,:] = summer(fast)
                sums[sidx,vidx,2,0,:] = summer(rc)
                sums[sidx,vidx,0,1,:] = summer(tob2)
                sums[sidx,vidx,1,1,:] = summer(fast2)
                sums[sidx,vidx,2,1,:] = summer(rc2)

                mask = (tob[:,0] > 0)
                mask2 = (tob2[:,0] > 0)
                inds = np.digitize(tob[mask,0], hist_bins)
                inds2 = np.digitize(tob2[mask2,0], hist_bins)  
                d = (tob[mask,:] - fast[mask,:])
                d2 = (tob2[mask2,:] - fast2[mask2,:])

                for tiss in range(2):
                    for b in range(hist_bins.size - 1):
                        diffs[sidx,vidx,0,tiss,b] = np.mean(d[inds == b+1,tiss])
                        diffs[sidx,vidx,1,tiss,b] = np.mean(d2[inds2 == b+1,tiss])

                # Tissue volume of each subcortical structure (Toblerone only)
                for stridx,struct in enumerate(all_structs):
                    for midx,meth in enumerate(['test', 'retest']):
                        tob_str = nibabel.load(op.join(ROOT, meth, sub, 'T1w', 'processed', 'tob_%s_0.7.nii.gz' % struct)).get_fdata()
                        structs[sidx,midx,stridx] = tob_str.sum()


        sio.savemat('HCP_data.mat', {
            'sums': sums, 'voxs': voxs, 'diffs': diffs, 'hist_bins': hist_bins, 'structs': structs
            })

if __name__ == "__main__":
    
    main(ROOT)