import os.path as op 
import os 
import subprocess
import multiprocessing as mp 
import copy 
import glob
import re
import sys 

import toblerone
from toblerone.utils import _loadFIRSTdir
from scipy.ndimage import binary_dilation
import numpy as np 
import regtricks as rt 
import nibabel as nib 
from regtricks.application_helpers import sum_array_blocks
from fsl.wrappers import bet
from sequence_params import SEQ_PARAMS
from svb import main 

root = '/mnt/hgfs/Data/maastricht_final'
outroot = op.join(root, 'processed')
CORES = 16
SCALES = range(1,3)
REFRESH = False

OXASL_PARAMS = (" --iaf=tc --cmethod=voxel --overwrite --no-report "
                " --debug --calib-aslreg ")


ASL_RAW_NAMES = {
    '01': 'acq_FAIR_run-',
    '03': 'task-ASL_acq-EP2DC32_dir-AP',
    '022': 'task-ASL_acq-EP2DC32_dir-AP',
    'SS': 'ASL_200vols_E00_M',
}

CALIB_RAW_NAMES = {
    '01': 'acq_FAIR_M0',
    '03' : 'task-M0_acq-EP2DC32_dir-AP',
    '022' : 'task-M0_acq-EP2DC32_dir-AP',
    'SS': 'M0_E00_M',
}

SUBS = list(ASL_RAW_NAMES.keys())

MCLFIRT_PARAMS = dict(smooth=0.125, scaling=0.25, stages=4, bins=512, rotation=0.25)
FLIRT_PARAMS = dict(dof=6, noresampblur=True, bins=1024, finesearch=4)
FOV_SHAPE = (120, 120, 30)


def shell(cmd):
    subprocess.run(cmd, shell=True)


def prepare_data():

    refresh = REFRESH

    for sub in SUBS: 

        os.makedirs(op.join(outroot, f'asl/sub-{sub}'), exist_ok=True)
        regdir = op.join(outroot, f'reg/sub-{sub}')
        asldir = op.join(outroot, f'asl/sub-{sub}')
        os.makedirs(regdir, exist_ok=True)

        calib_raw = glob.glob(op.join(root, 
            f'sub-{sub}', f'*{CALIB_RAW_NAMES[sub]}*.nii.gz'))[0]
        anat_dir = op.join(root, 
            f'sub-{sub}/sub-{sub}.anat')
        struct = op.join(anat_dir, 'T1_biascorr_brain.nii.gz')
        t1_spc = rt.ImageSpace(struct)

        # Motion correct calibration
        calib_spc = rt.ImageSpace(calib_raw)
        calib_mc_dir = f'{regdir}/calib_mc.mat'
        if not op.isdir(calib_mc_dir) or refresh:
            calib_mc = rt.mcflirt(calib_raw, refvol=0, **MCLFIRT_PARAMS)
            calib_mc.save_txt(calib_mc_dir)
        else: 
            calib_mc = rt.MotionCorrection.from_mcflirt(calib_mc_dir, calib_spc, calib_spc)
        calib_arr = nib.load(calib_raw).get_data()
        if len(calib_arr.shape) < 4: 
            calib_arr = calib_arr[...,None]

        # Save vol 0 for registration, mean across the remainder for final calib map 
        calib_vol0 = f'{asldir}/M0_AP_vol0.nii.gz'
        calib_spc.save_image(calib_arr[...,0], calib_vol0)
        calib_bet =  f'{asldir}/M0_AP_vol0_bet.nii.gz'
        # calib_mask = f'{asldir}/M0_AP_vol0_bet_mask.nii.gz'
        bet(calib_vol0, calib_bet, fracintensity=0.2, mask=True)

        # FLIRT calibration to structural, then take the inverse 
        calib2struct_mat = f'{regdir}/calib2struct.mat'
        if not op.exists(calib2struct_mat) or refresh:
            calib2struct = rt.flirt(calib_bet, struct, omat=calib2struct_mat, 
                cost='normmi', **FLIRT_PARAMS)
        else: 
            calib2struct = rt.Registration.from_flirt(calib2struct_mat, calib_spc, struct)
        struct2calib = calib2struct.inverse()

        # calib_struct = calib2struct.apply_to_array(calib_arr[...,0], calib_spc, struct)
        # rt.ImageSpace.save_like(struct, calib_struct, op.join(outroot, 'calib_struct.nii.gz'))

        # brain mask 
        bmask_array = nib.load(op.join(anat_dir, 'T1_biascorr_brain_mask.nii.gz')).get_data()
        bmask_array = binary_dilation(bmask_array)

        asls = sorted(glob.glob(op.join(root, 
                f'sub-{sub}', f'*{ASL_RAW_NAMES[sub]}*.nii.gz')))
        asls = [ a for a in asls if not a.count('_mcf.nii.gz') ]

        # Process each run of ASL 
        for r, asl in enumerate(asls):

            # Extract volume 0 of ASL, and reduce FoV of slices so that the 
            # 1.5, 3.0 and 4.5mm images cover same extents 
            run = r+1
            asl_spc = rt.ImageSpace(asl)
            asl_reduced_spc = asl_spc.resize([4,4,1], (120, 120, 30))
            asl_raw = nib.load(asl).get_data() 
            asl_vol0 = f'{asldir}/sub-{sub}_run-{run}_asl_vol0.nii.gz'
            calib_spc.save_image(asl_raw[...,0], asl_vol0)

            # FLIRT the calib to vol0 of the ASL 
            asl_vol0 = copy.deepcopy(asl_vol0)
            calib2run = f'{regdir}/run{run}_to_calib.mat'
            if not op.exists(calib2run) or refresh:
                calib2run = rt.flirt(calib_vol0, ref=asl_vol0, omat=calib2run, **FLIRT_PARAMS)
            else: 
                calib2run = rt.Registration.from_flirt(calib2run, asl_vol0, calib_vol0)

            # calib_asl = calib2run.apply_to_image(asl_vol0, asl_spc)
            # nib.save(calib_asl, op.join(outroot, 'calib_asl.nii.gz'))

            # Motion correct ASL series 
            asl_mc_dir = f'{regdir}/run-{run}.mat'
            if not op.exists(asl_mc_dir) or refresh:
                asl_mc = rt.mcflirt(asl, refvol=0, **MCLFIRT_PARAMS)
                asl_mc.save_txt(asl_mc_dir)
            else: 
                asl_mc = rt.MotionCorrection.from_mcflirt(asl_mc_dir, asl_spc, asl_spc)

            calib_transform = rt.chain(calib_mc, calib2run)
            struct_transform = rt.chain(struct2calib, calib2run)
            asl_transform = asl_mc
            out_spc = asl_reduced_spc
            fov_array = np.ones(out_spc.size, dtype=bool)
            struct_in_asl_path = f'{asldir}/sub-{sub}_run-{run}_struct.nii.gz'

            struct_in_asl = struct_transform.apply_to_image(struct, struct, order=1)
            nib.save(struct_in_asl, struct_in_asl_path)


            for scale in SCALES: 

                # Create scaled version of the space
                scale_spc = out_spc.resize_voxels(scale)
                fov_mask = (rt.Registration.identity().apply_to_array(
                        fov_array, out_spc, scale_spc, order=1) > 0.99)
  
                # Map the calibration into asl space    
                calib_arr_prepared = calib_transform.apply_to_array(
                                        calib_arr, calib_spc, scale_spc, order=1) 
                # calib_arr_prepared = calib_arr_prepared[fov_mask,1:].reshape(*FOV_SHAPE,-1)
                calib_motion_mask = (calib_arr_prepared > 0).all(-1)
                if calib_arr_prepared.ndim == 3: 
                    calib_arr_prepared = np.where(calib_motion_mask[...,None], 
                                        calib_arr_prepared, 0)
                else: 
                    calib_arr_prepared = np.where(calib_motion_mask[...,None], 
                                        calib_arr_prepared[...,1:], 0).mean(-1)

                asl_arr_prepared = asl_transform.apply_to_array(asl_raw, 
                                        asl, scale_spc, order=1)
                # asl_arr_prepared = asl_arr_prepared[fov_mask,:].reshape(*FOV_SHAPE,-1)
                asl_motion_mask = (asl_arr_prepared > 0).all(-1)
                asl_arr_prepared = np.where(asl_motion_mask[...,None], asl_arr_prepared, 0)

                print(f'sub {sub} run {run} scale {scale} space')
                asl_outpath = get_asl_path(sub, run, scale)
                print(asl_outpath, end='\n')

                calib_outpath = get_calib_path(sub, run, scale)
                scale_spc.save_image(calib_arr_prepared, calib_outpath)

                # transform mask from anat dir into ASL space
                bmask_asl = (struct_transform.apply_to_array(bmask_array, 
                    t1_spc, scale_spc, order=1) > 0.95)
                asl_mask = (asl_arr_prepared > 0).all(-1)
                final_mask = ((bmask_asl & asl_mask) & fov_mask)
            
                # slice_mask_prepared = (asl2struct.apply_to_array(slice_mask, calib_spc, scale_spc, order=1) > 0.9)
                # asl_mask = np.logical_and(slice_mask_prepared, bmask_asl) 
                scale_spc.save_image(asl_arr_prepared, asl_outpath)
                scale_spc.save_image(final_mask, get_mask_path(sub, 
                                    run, scale))

                struct2asl_path = get_struct2asl_path(sub, run, scale)
                struct_transform.save_fsl(struct2asl_path, t1_spc, scale_spc)


def subject_asl_runs(sub, scale):

    asls = os.listdir(op.join(outroot, 'asl', f'sub-{sub}'))
    asls = [ a for a in asls if 
        re.match(f'sub-{sub}_run-\d\d_scale-{scale}_asl.nii.gz', a) ]
    asls = [ op.join(outroot, 'asl', f'sub-{sub}', a) for a in asls ]
    return sorted(asls) 

def get_pv_path(sub, run, scale, method):
    gm, wm = [ op.join(outroot,
        f'asl/sub-{sub}/sub-{sub}_run-{run}_'
            + f'scale-{scale}_pv{tiss}_{method}.nii.gz')
            for tiss in ['gm', 'wm'] ] 
    return gm, wm 

def get_struct2asl_path(sub, run, scale):
    return op.join(outroot, 
        f'reg/sub-{sub}/struct_2_run-{run}_scale-{scale}.mat')


def prepare_projectors():

    debug = False
    identity = rt.Registration.identity()
    VOXPAD = 9 
    cores = 1 if debug else CORES

    for sub in SUBS:
        fslanat = op.join(root, f'sub-{sub}/sub-{sub}.anat')
        fsdir = op.join(fslanat, 'fs/surf')
        firstdir = op.join(fslanat, 'first_results')
        t1_spc = rt.ImageSpace(op.join(fslanat, 'T1.nii.gz'))

        for scale in SCALES:
            print(f'sub {sub} scale {scale}')

            # Do native space runs (using struct2asl transform)
            for rpt, _ in enumerate(subject_asl_runs(sub, scale)):
                rpt = rpt + 1
                native_spc = rt.ImageSpace(get_mask_path(sub, rpt, scale))

                struct2run = rt.Registration.from_flirt(
                    get_struct2asl_path(sub, rpt, scale), t1_spc, native_spc)

                lhemi = toblerone.Hemisphere(op.join(fsdir, 'lh.white'), 
                                             op.join(fsdir, 'lh.pial'), side='L')
                rhemi = toblerone.Hemisphere(op.join(fsdir, 'rh.white'), 
                                             op.join(fsdir, 'rh.pial'), side='R')

                hemis = [ lhemi.transform(struct2run.src2ref), rhemi.transform(struct2run.src2ref) ]

                rois = _loadFIRSTdir(firstdir)
                rois.pop('BrStem')
                for k,v in rois.items(): 
                    s = toblerone.Surface(v)
                    s = s.transform(t1_spc.FSL2world)
                    s = s.transform(struct2run.src2ref)
                    rois[k] = s

                proj = toblerone.Projector(hemis, native_spc, rois)
                fpath = op.join(outroot, 'asl', 
                            f'sub-{sub}/sub-{sub}_run-{rpt:02d}_scale-{scale}_pvs',
                            'projector.h5')
                proj.save(fpath)

def prepare_pvs():

    debug = False
    identity = rt.Registration.identity()
    VOXPAD = 9 
    cores = 1 if debug else CORES

    for sub in SUBS:
        fslanat = op.join(root, f'sub-{sub}/sub-{sub}.anat')
        fsdir = op.join(fslanat, 'fs')
        t1_spc = rt.ImageSpace(op.join(fslanat, 'T1.nii.gz'))
        fast_pvs = np.stack([ 
            nib.load(op.join(fslanat, 'T1_fast_pve_1.nii.gz')).get_fdata(),
            nib.load(op.join(fslanat, 'T1_fast_pve_2.nii.gz')).get_fdata() ], axis=-1)

        for scale in SCALES:
            print(f'sub {sub} scale {scale}')

            # Do native space runs (using struct2asl transform)
            for rpt, _ in enumerate(subject_asl_runs(sub, scale)):
                rpt = rpt + 1
                native_spc = rt.ImageSpace(get_mask_path(sub, rpt, scale))
                native_spc_expanded = native_spc.resize([0,0,-int(VOXPAD//scale)], 
                                                        native_spc.size+[0,0,2*int(VOXPAD//scale)])

                struct2run = rt.Registration.from_flirt(
                    get_struct2asl_path(sub, rpt, scale), t1_spc, native_spc)

                fast_pvs_native = struct2run.apply_to_array(fast_pvs, t1_spc, native_spc_expanded,
                    order=1)
                for fidx, tiss in enumerate(['fastgm', 'fastwm']):
                    fpath = op.join(outroot, 'asl', 
                            f'sub-{sub}/sub-{sub}_run-{rpt:02d}_scale-{scale}_pvs',
                            f'{tiss}.nii.gz')
                    os.makedirs(op.dirname(fpath), exist_ok=True)
                    fnative = fast_pvs_native[:,:,int(VOXPAD//scale):-int(VOXPAD//scale),fidx]
                    native_spc.save_image(fnative, fpath)

                tob_native = toblerone.pvestimation.complete(native_spc_expanded, struct2run, 
                    fslanat=fslanat, fsdir=fsdir, ones=False, cores=cores)   
                for key,img in tob_native.items():
                    tpath = op.join(outroot, 'asl', 
                            f'sub-{sub}/sub-{sub}_run-{rpt:02d}_scale-{scale}_pvs',
                            f'{key}.nii.gz')
                    tnative = img[:,:,int(VOXPAD//scale):-int(VOXPAD//scale)]
                    native_spc.save_image(tnative, tpath)

def get_asl_path(sub, run, scale):
    return op.join(outroot, 
        f'asl/sub-{sub}/sub-{sub}_run-{run:02d}_scale-{scale}_asl.nii.gz')

def get_mask_path(sub, run, scale):
    return op.join(outroot, 
        f'asl/sub-{sub}/sub-{sub}_run-{run:02d}_scale-{scale}_mask.nii.gz')

def get_calib_path(sub, run, scale):
    return op.join(outroot,
        f'asl/sub-{sub}/sub-{sub}_run-{run:02d}_scale-{scale}_M0_acq-AP.nii.gz')

def run_oxasl():

    jobs = []

    for sub in SUBS: 
        os.makedirs(op.join(outroot, f'oxasl/sub-{sub}'), exist_ok=True)

        for scale in SCALES: 
            for run, asl in enumerate(subject_asl_runs(sub, scale)):
                run = run + 1 
                oxasl_out = op.join(outroot, 
                    f'oxasl/sub-{sub}/sub-{sub}_run-{run:02d}_scale-{scale}')
                if not op.exists(oxasl_out) or True:
                    
                    os.makedirs(oxasl_out, exist_ok=True)
                    calib = get_calib_path(sub, run, scale)
                    # fsl_anat = op.join(root, 
                    #     f'sub-{sub}/sub-{sub}.anat')
                    # fsdir = op.join(root, 
                    #     f'sub-{sub}/sub-{sub}-fs')
                    mask = get_mask_path(sub, run, scale)
                    cmd = 'oxasl -i {} -c {} -o {} -m {} '.format(asl, calib, oxasl_out, mask)

                    params = copy.deepcopy(SEQ_PARAMS)
                    slicedt = params.pop('slicedt')        
                    cmd += " ".join([ f'--{k}={v}' for k,v in params.items() ])
                    cmd += OXASL_PARAMS + f' --slicedt={scale*slicedt}'
                    jobs.append(cmd)

    with mp.Pool(CORES) as p:
        p.map(shell, jobs)


def run_svb_comparison():

    jobs = []

    for sub in SUBS: 
        os.makedirs(op.join(outroot, f'oxasl/sub-{sub}'), exist_ok=True)

        for scale in SCALES: 
            for run, asl in enumerate(subject_asl_runs(sub, scale)):
                run = run + 1 
                oxasl_out = op.join(outroot, 
                    f'oxasl/sub-{sub}/sub-{sub}_run-{run:02d}_scale-{scale}_oxasl')
                    
                os.makedirs(oxasl_out, exist_ok=True)
                calib = get_calib_path(sub, run, scale)
                mask = get_mask_path(sub, run, scale)
                cmd = 'oxasl -i {} -c {} -o {} -m {} '.format(asl, calib, oxasl_out, mask)

                pvgm = op.join(outroot, 'asl', 
                        f'sub-{sub}/sub-{sub}_run-{run:02d}_scale-{scale}_pvs',
                        f'fastgm.nii.gz')
                pvwm = op.join(outroot, 'asl', 
                        f'sub-{sub}/sub-{sub}_run-{run:02d}_scale-{scale}_pvs',
                        f'fastwm.nii.gz')

                params = copy.deepcopy(SEQ_PARAMS)
                slicedt = params.pop('slicedt')        
                cmd += " ".join([ f'--{k}={v}' for k,v in params.items() ])
                cmd += OXASL_PARAMS + f' --slicedt={scale*slicedt} '
                cmd += f'--pvcorr --pvgm {pvgm} --pvwm {pvwm}'
                jobs.append(cmd)

                svb_dir = op.join(outroot, 
                    f'oxasl/sub-{sub}/sub-{sub}_run-{run:02d}_scale-{scale}_svb')
                projector = toblerone.Projector.load(
                    op.join(outroot, 'asl', 
                        f'sub-{sub}/sub-{sub}_run-{run:02d}_scale-{scale}_pvs',
                        'projector.h5')
                )
                svb_opts = {
                    "mode": "hybrid", 
                    "learning_rate": 0.5,
                    "outformat": ['nii', 'gii', 'cii'], 
                    "batch_size" : 10,
                    "sample_size" : 5,
                    "epochs" : 750,
                    "log_stream" : sys.stdout,
                    "display_step": 10,  
                    "projector": projector,
                    "casl": True, 
                    "prior_type": "M",
                    "output": svb_dir, 
                    "mask": mask,
                    
                    'casl': False, 
                    'tis': [1.8],
                    't1b': 2.2, 
                    'alpha': 0.95,
                    'slicedt': 0.03197,
                    'bolus': 0.7, 
                    'repeats': 100, 
                }

                runtime, svb, training_history = main.run(
                    asl, "aslrest",
                    **svb_opts)


    with mp.Pool(CORES) as p:
        p.map(shell, jobs)

if __name__ == "__main__":
    
    # prepare_data()
    # prepare_pvs()
    # run_oxasl()

    # prepare_projectors()
    run_svb_comparison()
