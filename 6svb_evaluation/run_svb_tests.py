import sys 
sys.path.append('../pvec_resampling')

import numpy as np 
import os.path as op 
import regtricks as rt 
import nibabel as nib
from svb.main import run 
from svb.data import HybridModel
import itertools
import functools
import multiprocessing
import subprocess
import os 
import toblerone 
from scipy import interpolate
import shutil
import fabber_funcs
from simulate_data import (CBF, ATT, TAU, PLD_REPEATS, SNR, N_VAR, CASL,
                            PLDS, N_REPEATS, make_transform)
import sys 
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except ImportError:
    import tensorflow as tf
from svb_models_asl import AslRestModel 
from svb.data import HybridModel

SIM_ROOT = '/mnt/hgfs/Data/thesis_data/svb_evaluation/sim_rpts'
CORES = 12 

Q2 = [1,10,100,1000]
# Q2 = [ 100 ]
ROI_CENTS = [ 20398, 6110, 11075, 13634, 15547, 18533, 17385 ]
N_DILATIONS = 12
SNR_LEVELS = [0, 2]


def svb_make_hybrid_data(pmean, pvar, snr, opts, proj):
    data = np.zeros((*proj.spc.size, opts['repeats'] * len(opts['plds'])))
    asl_model = AslRestModel(
            HybridModel(proj.spc.make_nifti(data), projector=proj), **opts)

    tpts = asl_model.tpts()
    nvox = int(proj.spc.size.prod())
    with tf.Session() as sess:
        cbf = np.concatenate([
            np.random.normal(pmean[0], pvar[0], size=[proj.n_surf_nodes, 1]), 
            np.random.normal(pmean[2], pvar[2], size=[nvox, 1]), 
            np.random.normal(pmean[0], pvar[0], size=[proj.n_subcortical_nodes, 1]), 
        ])
        att = np.concatenate([
            np.random.normal(pmean[1], pvar[1], size=[proj.n_surf_nodes, 1]), 
            np.random.normal(pmean[3], pvar[3], size=[nvox, 1]), 
            np.random.normal(pmean[1], pvar[1], size=[proj.n_subcortical_nodes, 1]), 
        ])
        data = sess.run(asl_model.evaluate(
             [ cbf.astype(np.float32), att.astype(np.float32) ], tpts))

    data = proj.node2vol(data, edge_scale=True)
    data = data.reshape(*proj.spc.size, tpts.shape[-1])
    nvar = N_VAR(snr, opts['repeats'])
    data += np.random.normal(0, nvar, size=data.shape)
    return data 

def run_svb(asl, mask, odir, opts, projector):
    
    options = {
        "mode": "hybrid", 
        "learning_rate": 0.5,
        "outformat": ['nii', 'gii', 'cii'], 
        "batch_size" : opts['repeats'],
        "sample_size" : 5,
        "epochs" : 1000,
        "log_stream" : sys.stdout,
        "display_step": 100,  
        "projector": projector,
        "casl": True, 
        "prior_type": "M",
        "output": odir, 
        "mask": mask,
        "save_param_history": True, 
        "save_cost_history": True, 
        "gamma_q2": 100, 
        **opts,

    }

    runtime, svb, training_history = run(
        asl, "aslrest",
        **options)


def rois_as_rings(surf): 
    rois = []
    for rc in ROI_CENTS: 
        last_neighbours = set([])
        neighbours = []
        vtx_neighbours = np.array(rc)
        for _ in range(N_DILATIONS): 
            tri_neighbours = np.isin(surf.tris, vtx_neighbours).any(-1)
            vtx_neighbours = surf.tris[tri_neighbours,:].flatten()
            new_neighbours = set(vtx_neighbours) - set(last_neighbours)
            neighbours.append(list(new_neighbours))
            last_neighbours = new_neighbours | last_neighbours
        rois.append(neighbours)
    return rois 

def rois_flat(surf):
    roi_rings = rois_as_rings(surf)
    return [ np.concatenate(rings) for rings in roi_rings ]



def simulate_repeat_data(): 

    # Load anatomical surfaces for subject 103818 of HCP 
    sdir = "/mnt/hgfs/Data/toblerone_evaluation_data/HCP_retest/test/103818/T1w/fsaverage_LR32k"
    LPS = op.join(sdir, '103818.L.pial.32k_fs_LR.surf.gii')
    LWS = op.join(sdir, '103818.L.white.32k_fs_LR.surf.gii')
    RPS = op.join(sdir, '103818.R.pial.32k_fs_LR.surf.gii')
    RWS = op.join(sdir, '103818.R.white.32k_fs_LR.surf.gii')
    LIS = op.join(sdir, '103818.L.very_inflated.32k_fs_LR.surf.gii')
    RIS = op.join(sdir, '103818.R.very_inflated.32k_fs_LR.surf.gii')
    spc = toblerone.ImageSpace(op.join(sdir, '../processed/fast_3.0.nii.gz'))

    # Load FIRST surfaces, pop brain stem 
    rois = toblerone.utils._loadFIRSTdir(op.join(sdir, '../processed/first'))
    rois.pop('BrStem')
    t1_spc = toblerone.ImageSpace(op.join(sdir, 
        '../processed/first/T1w_acpc_dc_restore_all_fast_firstseg.nii.gz'))
    for k,v in rois.items(): 
        s = toblerone.Surface(v)
        s = s.transform(t1_spc.FSL2world)
        rois[k] = s 

    # Produce the common space projector 
    lhemi = toblerone.Hemisphere(LWS, LPS, 'L')
    rhemi = toblerone.Hemisphere(RWS, RPS, 'R')
    proj_path = op.join(SIM_ROOT, 'projector_common.h5')
    if not op.exists(proj_path):
        proj = toblerone.Projector([lhemi, rhemi], spc, rois)
        proj.save(proj_path)
    else: 
        proj = toblerone.Projector.load(proj_path)

    # Define ground truth parameter maps 
    # WM will be constant 
    # GM will have varying CBF and ATT. These are generated by 
    # producing a sine field and using that to modulate the distributions 
    shutil.copy(LIS, op.join(SIM_ROOT, op.split(LIS)[1]))
    shutil.copy(RIS, op.join(SIM_ROOT, op.split(RIS)[1]))
    LIS = toblerone.Surface(LIS)
    RIS = toblerone.Surface(RIS)

    activation = np.zeros(LIS.n_points)
    roi_rings = rois_as_rings(LIS)
    for rings, sign in zip(roi_rings, [1,1,1,1,1,-1,-1]): 
        x = np.arange(len(rings))
        for intensity, n in zip(1 - (x / N_DILATIONS) ** 3,
                                rings): 
            activation[n] = sign * intensity

    ctx_cbf = 60 + (25 * activation)
    LIS.save_metric(ctx_cbf, 
                    op.join(SIM_ROOT, 'L_cortex_cbf_truth.func.gii'))
    RIS.save_metric(ctx_cbf, 
                    op.join(SIM_ROOT, 'R_cortex_cbf_truth.func.gii'))
    ctx_cbf = np.concatenate([ctx_cbf, ctx_cbf])
    ctx_cbf_vol = proj.surf2vol(ctx_cbf, edge_scale=False)
    spc.save_image(ctx_cbf_vol, op.join(SIM_ROOT, 'cortex_cbf_truth.nii.gz'))
    ctx_att = 1.3 * np.ones_like(ctx_cbf)

    # Simulating function is hardcoded to use the cortex parameters and 
    # reference subcortex values 
    def simulator(projector, snr):
        opts = dict(plds=PLDS, repeats=PLD_REPEATS, tau=TAU, casl=CASL)
        data = np.zeros((*projector.spc.size, 
                         opts['repeats'] * len(opts['plds'])))
        asl_model = AslRestModel(
                HybridModel(projector.spc.make_nifti(data), 
                            projector=projector), **opts)

        tpts = asl_model.tpts()
        nvox = int(projector.spc.size.prod())
        with tf.Session() as sess:
            cbf = np.concatenate([
                    ctx_cbf[:,None],
                    20 * np.ones([nvox, 1]), 
                    60 * np.ones([projector.n_subcortical_nodes, 1]), 
            ])
            att = np.concatenate([
                    ctx_att[:,None],
                    1.6 * np.ones([nvox, 1]), 
                    1.3 * np.ones([projector.n_subcortical_nodes, 1]), 
            ])
            data = sess.run(asl_model.evaluate(
                 [ cbf.astype(np.float32), att.astype(np.float32) ], tpts))

        data = projector.node2vol(data, edge_scale=True)
        data = data.reshape(*projector.spc.size, tpts.shape[-1])
        nvar = N_VAR(snr, opts['repeats'])
        data += np.random.normal(0, nvar, size=data.shape)
        return data 

    for rpt, snr in itertools.product(range(N_REPEATS), SNR_LEVELS):

        if snr: 
            odir = op.join(SIM_ROOT, f'rpt{rpt}_hisnr')
        else: 
            odir = op.join(SIM_ROOT, f'rpt{rpt}_losnr')

        os.makedirs(odir, exist_ok=True)

        # Produce random transform if it doesn't already exist 
        tpath = op.join(odir, 'cmn2native.txt')
        if not op.exists(tpath): 
            trans = make_transform()
            trans.save_txt(tpath)
        else: 
            trans = rt.Registration(tpath)

        # FIRST surfaces in native space 
        rois = toblerone.utils._loadFIRSTdir(op.join(sdir, '../processed/first'))
        rois.pop('BrStem')
        t1_spc = toblerone.ImageSpace(op.join(sdir, 
            '../processed/first/T1w_acpc_dc_restore_all_fast_firstseg.nii.gz'))
        for k,v in rois.items(): 
            s = toblerone.Surface(v)
            s = s.transform(t1_spc.FSL2world)
            s = s.transform(trans.src2ref)
            rois[k] = s 

        # Native space projector 
        lhemi = toblerone.Hemisphere(LWS, LPS, 'L')
        rhemi = toblerone.Hemisphere(RWS, RPS, 'R')
        proj_path = op.join(odir, f'projector_rpt{rpt}.h5')
        if not op.exists(proj_path):
            proj = toblerone.Projector([lhemi.transform(trans.src2ref),
                                        rhemi.transform(trans.src2ref)], 
                                       spc, rois)
            proj.save(proj_path)
        else: 
            proj = toblerone.Projector.load(proj_path)

        # Native space mask and pvs 
        pvs_native = proj.pvs()
        mask_native = (pvs_native[...,:2] > 0.05).any(-1)
        spc.save_image(pvs_native[...,0], op.join(odir, 'pvgm_native.nii.gz'))
        spc.save_image(pvs_native[...,1], op.join(odir, 'pvwm_native.nii.gz'))
        spc.save_image(mask_native, op.join(odir, 'mask_native.nii.gz'))

        # Common space mask and pvs 
        pvs_common = trans.inverse().apply_to_array(pvs_native, spc, spc, order=1)
        mask_common = (pvs_common[...,:2] > 0.05).any(-1)
        spc.save_image(pvs_common[...,0], op.join(odir, 'pvgm_common.nii.gz'))
        spc.save_image(pvs_common[...,1], op.join(odir, 'pvwm_common.nii.gz'))
        spc.save_image(mask_common, op.join(odir, 'mask_common.nii.gz'))

        # Generate native space data 
        data = simulator(proj, SNR + (snr * SNR))
        ndatap = op.join(odir, f'asl_rpt{rpt}_native.nii.gz')
        spc.save_image(data, ndatap)

        # Transform into commonn space 
        cdatap = op.join(odir, f'asl_rpt{rpt}_common.nii.gz')
        cdata = trans.inverse().apply_to_array(data, spc, spc, order=1)
        spc.save_image(cdata, cdatap)


def evaluate_spatial_prior(): 

    rpts = 4
    basil_opts = {    
        "repeats": rpts, 
        "casl": True, 
        "tau": TAU,
        "bat": ATT[0], 
        "batwm": ATT[1],
        ** { f'pld{n+1}' : p for n,p in enumerate(PLDS) }
    }

    root = op.join(SIM_ROOT, '../spatial_prior_tests')
    os.makedirs(root, exist_ok=True)
    proj = toblerone.Projector.load(op.join(root, 'L_hemi_projector.h5'))

    pv_path = op.join(root, 'pvs.nii.gz')
    mask_path = op.join(root, 'mask.nii.gz')
    pvs = proj.pvs()
    mask = pvs[...,:2].any(-1)
    proj.spc.save_image(pvs[...,:2], pv_path)
    proj.spc.save_image(mask, mask_path)

    opts = dict(plds=PLDS, repeats=rpts, tau=TAU, casl=CASL)

    # constant parameters, high noise
    pmean = [CBF[0], ATT[0], CBF[1], ATT[1]]
    pvar = [0,0,0,0]
    adata_path = op.join(root, 'asl_acquisition_noise.nii.gz')
    data = svb_make_hybrid_data(pmean, pvar, 3*SNR, opts, proj)
    proj.spc.save_image(data, adata_path)

    # variable parameters, low noise 
    pvar = [8, 0.15, 3, 0.25]
    pdata_path = op.join(root, 'asl_param_noise.nii.gz')
    data = svb_make_hybrid_data(pmean, pvar, 10*SNR, opts, proj)
    proj.spc.save_image(data, pdata_path)

    # BASIL for both of them 
    odir = op.join(root, 'basil_acquisition_noise')
    basil_jobs = [ 
        fabber_funcs.basil_cmd(adata_path, mask_path, basil_opts, odir, pv_path)
    ]

    odir = op.join(root, 'basil_parameter_noise')
    basil_jobs.append(
        fabber_funcs.basil_cmd(pdata_path, mask_path, basil_opts, odir, pv_path)
    )

    for q2 in Q2:

        odir = op.join(root, f'svb_acquisition_noise_q2-{q2}')
        svb_opts = dict(gamma_q2=q2, epochs=750, **opts)
        run_svb(adata_path, mask_path, odir, svb_opts, proj)

        odir = op.join(root, f'svb_parameter_noise_q2-{q2}')
        run_svb(pdata_path, mask_path, odir, svb_opts, proj)


    # Part 2: a sine varying cbf map on the surface 
    inds = np.indices(proj.spc.size)
    LIS = toblerone.Surface(
        op.join(SIM_ROOT, '103818.L.very_inflated.32k_fs_LR.surf.gii'))
    LIS = LIS.transform(proj.spc.world2vox)
    scale = 3
    sine = (np.sin(inds[1] / scale) 
            + np.sin(inds[0] / scale) 
            + np.sin(inds[2] / scale))
    sine = sine / sine.max()

    ctx_sine = interpolate.interpn(
            points=[ np.arange(d) for d in proj.spc.size ], 
            values=sine, 
            xi=LIS.points
        )    

    ctx_cbf = CBF[0] + (25 * ctx_sine)
    ctx_att = ATT[0] * np.ones_like(ctx_cbf)
    LIS.save_metric(ctx_cbf, op.join(root, 'L_ctx_sine.func.gii'))

    data_model = HybridModel(proj.spc.make_nifti(data), 
                projector=proj)
    asl_model = AslRestModel(data_model, casl=CASL,
                plds=PLDS, repeats=rpts, tau=TAU)
    tpts = asl_model.tpts()
    nvox = proj.spc.size.prod()

    with tf.Session() as sess:
        cbf = np.concatenate([
                ctx_cbf[:,None],
                CBF[1] * np.ones([nvox, 1]), 
        ])
        att = np.concatenate([
                ctx_att[:,None],
                ATT[1] * np.ones([nvox, 1]), 
        ])
        data = sess.run(asl_model.evaluate(
                [ cbf.astype(np.float32), att.astype(np.float32) ], tpts))

    data = proj.node2vol(data, edge_scale=True).reshape(*proj.spc.size, -1)
    nvar = N_VAR(2 * SNR, rpts)
    data[mask,:] += np.random.normal(0, nvar, size=data[mask,:].shape)
    data_path = op.join(root, 'L_ctx_sine_simdata.nii.gz')
    proj.spc.save_image(data, data_path)

    for in_weight in Q2: 
        odir = op.join(root, f'svb_sine_inweight_{in_weight}')
        svb_opts = dict(plds=PLDS, repeats=rpts, tau=TAU,
            laplacian_in_weight=in_weight, epochs=750)
        run_svb(data_path, mask_path, odir, svb_opts, proj)

    worker = functools.partial(subprocess.run, shell=True)
    with multiprocessing.Pool(CORES) as p: 
        p.map(worker, basil_jobs)



def fit_simulated_repeats(): 

    # simulations 
    sim_root = op.join(SIM_ROOT, '../hybrid_svb_sims')
    svb_jobs = []
    oxasl_jobs = [] 

    basil_opts = {    
        "repeats": PLD_REPEATS, 
        "casl": True, 
        "tau": TAU,
        "bat": ATT[0], 
        "batwm": ATT[1], 
        **{ f'pld{n+1}' : p for n,p in enumerate(PLDS) }
    }

    svb_opts = dict(plds=PLDS, repeats=PLD_REPEATS, tau=TAU, casl=True)

    # # SVB on common space data 
    # rdir =  op.join(sim_root, 'rpt1')
    # nmask = op.join(rdir, 'mask_common.nii.gz')
    # ndata = op.join(rdir, 'asl_rpt1_common.nii.gz')
    # proj = toblerone.Projector.load(op.join(sim_root, 'projector_common.h5'))
    # odir = op.join(sim_root, 'svb_common')
    # svb_jobs.append([ndata, nmask, odir, svb_opts, proj])

    for rpt, snr in itertools.product(range(N_REPEATS), SNR_LEVELS):

        if snr: 
            rdir = op.join(sim_root, f'rpt{rpt}_hisnr')
        else: 
            rdir = op.join(sim_root, f'rpt{rpt}_losnr')

        # SVB on native space data 
        nmask = op.join(rdir, 'mask_native.nii.gz')
        ndata = op.join(rdir, f'asl_rpt{rpt}_native.nii.gz')
        proj = toblerone.Projector.load(op.join(rdir, f'projector_rpt{rpt}.h5'))
        odir = op.join(rdir, 'svb_native')
        svb_jobs.append([ndata, nmask, odir, svb_opts, proj])

        # BASIL on native space 
        pvs = [ op.join(rdir, 'pvgm_native.nii.gz'), 
                op.join(rdir, 'pvwm_native.nii.gz') ] 
        odir = op.join(rdir, 'basil_native')
        oxasl_jobs.append(fabber_funcs.basil_cmd(ndata, nmask, basil_opts, odir, pvs))

        # # LR on native space 
        # odir = op.join(rdir, 'lr_native')
        # oxasl_jobs += lr_method.basil_lr(ndata, nmask, basil_opts, odir, pvs)

        # BASIL on common space 
        cmask = op.join(rdir, 'mask_common.nii.gz')
        cdata = op.join(rdir, f'asl_rpt{rpt}_common.nii.gz')
        pvs = [ op.join(rdir, 'pvgm_common.nii.gz'), 
                op.join(rdir, 'pvwm_common.nii.gz') ] 
        odir = op.join(rdir, 'basil_common')
        oxasl_jobs.append(fabber_funcs.basil_cmd(cdata, cmask, basil_opts, odir, pvs))

        # # LR on common space 
        # odir = op.join(rdir, 'lr_common')
        # oxasl_jobs += lr_method.basil_lr(cdata, cmask, basil_opts, odir, pvs)

    for job in svb_jobs: 
        run_svb(*job)

    worker = functools.partial(subprocess.run, shell=True)
    with multiprocessing.Pool(CORES) as p: 
        p.map(worker, oxasl_jobs)

    for rpt, snr in itertools.product(range(N_REPEATS), SNR_LEVELS):

        if snr: 
            rdir = op.join(sim_root, f'rpt{rpt}_hisnr')
        else: 
            rdir = op.join(sim_root, f'rpt{rpt}_losnr')

        proj = toblerone.Projector.load(op.join(rdir, f'projector_rpt{rpt}.h5'))
        basil = nib.load(op.join(rdir, 'basil_native/basil_out/step2/mean_ftiss.nii.gz')).get_fdata()
        basil_on_surf = proj.vol2surf(basil.flatten(), edge_scale=False)
        lps = proj['LPS']
        rps = proj['RPS']
        lps.save_metric(basil_on_surf[lps.n_points:], 
                        op.join(rdir, 'basil_native/L_cortex_ftiss.func.gii'))
        rps.save_metric(basil_on_surf[-rps.n_points:], 
                        op.join(rdir, 'basil_native/R_cortex_ftiss.func.gii'))



if __name__ == '__main__':

    evaluate_spatial_prior()
    # simulate_repeat_data()
    # fit_simulated_repeats()
