import numpy as np 
import regtricks as rt 
import nibabel as nib
from scipy.spatial.transform import Rotation
import lr_method
import pathlib
import fabber_funcs
import functools
import multiprocessing
import subprocess

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except ImportError:
    import tensorflow as tf
from svb_models_asl import AslRestModel 
from svb.data import VolumetricModel


SIM_ROOT = pathlib.Path('/mnt/hgfs/Data/thesis_data/pvec_resampling/simulated')
REAL_ROOT = pathlib.Path('/mnt/hgfs/Data/thesis_data/pvec_resampling/maastricht')

CBF = [60, 20]
ATT = [1.3, 1.6]
CASL = True
TAU = 1.8
PLD_REPEATS = 8
PLDS = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
N_REPEATS = 4
SIGMAX = 42
SNR = 9
def N_VAR(snr, rpts):
    return SIGMAX * np.sqrt(len(PLDS) * rpts) / snr


def make_transform():
    """
    Generate a random affine transformation: a small rotation around
    z and x axis, with a small translation in xyz. 
    """
    
    rot1 = Rotation.from_euler('z', np.random.randint(0,5), degrees=True)
    rot2 = Rotation.from_euler('x', np.random.randint(0,5), degrees=True)
    transform = np.eye(4)
    transform[:3,:3] = (rot1.as_matrix() @ rot2.as_matrix())
    transform[:3,3] = 6 * np.random.rand(3)
    return rt.Registration(transform)


def svb_sim_data(params, snr, opts, shape):
    spc = rt.ImageSpace.create_axis_aligned([0,0,0], shape, [1,1,1])
    data = np.zeros((*spc.size, opts['repeats'] * len(opts['plds'])))
    asl_model = AslRestModel(
            VolumetricModel(spc.make_nifti(data)), **opts)

    tpts = asl_model.tpts()
    with tf.Session() as sess:
        ones = np.ones([asl_model.data_model.n_nodes, 1], dtype=np.float32)
        data = sess.run(asl_model.evaluate(
            [ p * ones for p in params], tpts))

    data = data.reshape(*spc.size, tpts.shape[-1])
    nvar = N_VAR(snr, opts['repeats'])
    data += np.random.normal(0, nvar, size=data.shape)
    return data 


def prepare_sim_data():
    
    pvs = nib.load(str(SIM_ROOT / 'tob_all_stacked_3.0.nii.gz'))
    spc = rt.ImageSpace(pvs)
    pvs = pvs.get_fdata()[...,:2]
    brain_mask = (pvs > 0.01).any(-1)

    opts = dict(plds=PLDS, repeats=PLD_REPEATS, tau=TAU, casl=CASL, 
                pvcorr=True, mask=brain_mask, pvgm=pvs[...,0],
                pvwm=pvs[...,1])
    data = svb_sim_data([CBF[0], ATT[0], CBF[1], ATT[1]], 3*SNR, opts, spc.size)
    spc.save_image(data, str(SIM_ROOT / 'asl_native.nii.gz'))
    spc.save_image(brain_mask, str(SIM_ROOT / 'mask.nii.gz'))

    trans = make_transform()
    data_t = trans.apply_to_array(data, spc, spc, order=1)
    data_t = trans.inverse().apply_to_array(data_t, spc, spc, order=1)
    pvs_t = trans.apply_to_array(pvs, spc, spc, order=1)
    pvs_t = trans.inverse().apply_to_array(pvs_t, spc, spc, order=1)
    spc.save_image(data_t, str(SIM_ROOT / 'asl_resampled.nii.gz'))
    spc.save_image(pvs_t, str(SIM_ROOT / 'pvs_resampled.nii.gz'))

    for idx in range(N_REPEATS):

        trans = make_transform()
        trans.save_txt(str(SIM_ROOT / f'rpt{idx+1}_common_to_acq.txt'))
        pvs_rpt = trans.apply_to_array(pvs, spc, spc, order=1)
        opts = dict(plds=PLDS, repeats=PLD_REPEATS, tau=TAU, casl=CASL, 
                    pvcorr=True, mask=brain_mask, pvgm=pvs_rpt[...,0],
                    pvwm=pvs_rpt[...,1])
        data_rpt = svb_sim_data([CBF[0], ATT[0], CBF[1], ATT[1]], 
                                SNR, opts, spc.size)
        spc.save_image(pvs_rpt, str(SIM_ROOT / f'pvs_rpt{idx}_in_acq.nii.gz'))
        spc.save_image(data_rpt, str(SIM_ROOT / f'asl_rpt{idx}_in_acq.nii.gz'))
        data_cmn = trans.inverse().apply_to_array(data_rpt, spc, spc, order=1)
        pvs_cmn = trans.inverse().apply_to_array(pvs_rpt, spc, spc, order=1)
        spc.save_image(data_cmn, str(SIM_ROOT / f'asl_rpt{idx}_in_common.nii.gz'))
        spc.save_image(pvs_cmn, str(SIM_ROOT / f'pvs_rpt{idx}_double_resampled.nii.gz'))

def fit_simulations(): 
    jobs = []
    opts = {    
        "repeats": PLD_REPEATS, 
        "casl": True, 
        "tau": TAU,
        "bat": ATT[0], 
        "batwm": ATT[1],
        ** { f'pld{n+1}' : p for n,p in enumerate(PLDS) }
    }

    # Native data, native PVs
    asl = str(SIM_ROOT / 'asl_native.nii.gz')
    mask = str(SIM_ROOT / 'mask.nii.gz')
    pvs_native = str(SIM_ROOT / 'tob_all_stacked_3.0.nii.gz')
    odir = str(SIM_ROOT / 'basil_native')
    jobs.append(fabber_funcs.basil_cmd(asl, mask, opts, odir, pvs_native))

    odir = str(SIM_ROOT / 'lr_native')
    jobs += lr_method.basil_lr(asl, mask, opts, odir, pvs_native)

    # Resampled data, native PVs
    asl = str(SIM_ROOT / 'asl_resampled.nii.gz')
    odir = str(SIM_ROOT / 'basil_naive')
    jobs.append(fabber_funcs.basil_cmd(asl, mask, opts, odir, pvs_native))

    odir = str(SIM_ROOT / 'lr_naive')
    jobs += lr_method.basil_lr(asl, mask, opts, odir, pvs_native)

    # Resampled data, resampled PVs
    pvs = str(SIM_ROOT / 'pvs_resampled.nii.gz')
    odir = str(SIM_ROOT / 'basil_double_resampled')
    jobs.append(fabber_funcs.basil_cmd(asl, mask, opts, odir, pvs))

    odir = str(SIM_ROOT / 'lr_double_resampled')
    jobs += lr_method.basil_lr(asl, mask, opts, odir, pvs)

    for idx in range(N_REPEATS):

        # Repeat data in common space, with double-resamp PVs
        asl = str(SIM_ROOT / f'asl_rpt{idx}_in_common.nii.gz')
        pvs = str(SIM_ROOT / f'pvs_rpt{idx}_double_resampled.nii.gz')
        odir = str(SIM_ROOT / f'basil_rpt{idx}_double_resampled')
        jobs.append(fabber_funcs.basil_cmd(asl, mask, opts, odir, pvs))

        odir = str(SIM_ROOT /  f'lr_rpt{idx}_double_resampled')
        jobs += lr_method.basil_lr(asl, mask, opts, odir, pvs)

        # Same data, with naive PVs 
        odir = str(SIM_ROOT / f'basil_rpt{idx}_naive')
        jobs.append(fabber_funcs.basil_cmd(asl, mask, opts, odir, pvs_native))

        odir = str(SIM_ROOT /  f'lr_rpt{idx}_naive')
        jobs += lr_method.basil_lr(asl, mask, opts, odir, pvs_native)

        # Repeats analysed in their native acquisition space 
        asl = str(SIM_ROOT / f'asl_rpt{idx}_in_acq.nii.gz')
        pvs = str(SIM_ROOT / f'pvs_rpt{idx}_in_acq.nii.gz')
        odir = str(SIM_ROOT / f'basil_rpt{idx}_in_acq')
        jobs.append(fabber_funcs.basil_cmd(asl, mask, opts, odir, pvs))

        odir = str(SIM_ROOT /  f'lr_rpt{idx}_in_acq')
        jobs += lr_method.basil_lr(asl, mask, opts, odir, pvs)

    worker = functools.partial(subprocess.run, shell=True)
    with multiprocessing.Pool(12) as p: 
        p.map(worker, jobs)



def prepare_real_data():

#     native space 
    wdir = str(REAL_ROOT / 'native')
    apath = f'{wdir}/sub-SS_run-01_scale-2_space-native_asl.nii.gz'
    cpath = f'{wdir}/sub-SS_run-01_scale-2_space-native_M0.nii.gz'
    ppaths = [f'{wdir}/sub-SS_run-01_scale-2_space-native_pvgm_tob.nii.gz',
          f'{wdir}/sub-SS_run-01_scale-2_space-native_pvwm_tob.nii.gz']

    asl = nib.load(apath)
    spc = rt.ImageSpace(apath)
    m0 = nib.load(cpath)

    transform = np.eye(4)
    transform[:3,3] = [ 1.3, -2.1, 0 ]
    transform = rt.Registration(transform)

    asl_trans = transform.apply_to_array(asl.get_fdata(), src=asl, ref=asl, order=1)
    asl_trans = transform.inverse().apply_to_array(asl_trans, src=asl, ref=asl, order=1)
    spc.save_image(asl_trans, f'{wdir}/asl_resamp.nii.gz')
    asl_trans = spc.make_nifti(asl_trans)

    m0_trans = transform.apply_to_image(m0, ref=asl, order=1)
    m0_trans = transform.inverse().apply_to_image(m0_trans, ref=asl, order=1)
    m0_trans.to_filename(f'{wdir}/m0_resamp.nii.gz')

    for tiss, path in zip(['pvgm', 'pvwm'], ppaths):
        pvtrans = transform.apply_to_image(path, ref=asl, order=1)
        pvtrans = transform.inverse().apply_to_image(pvtrans, ref=asl, order=1)
        pvtrans.to_filename(f'{wdir}/{tiss}_resamp.nii.gz')

    tivol = np.indices(spc.size, dtype=np.float32)[-1]
    tivol *= (2 * 0.0325)
    tivol += 1.8
    tivol = np.stack(200 * [tivol], axis=-1)
    tivol_path = f'{wdir}/tivol.nii.gz'
    spc.save_image(tivol, tivol_path)

    # common space 
    wdir = str(REAL_ROOT / 'common')
    a2c = rt.Registration.from_flirt(f'{wdir}/run1_to_calib.mat',
        f'{wdir}/sub-SS_run-1_asl_vol0.nii.gz', 
        f'{wdir}/M0_AP_vol0.nii.gz'
    )

    c2s = rt.Registration.from_flirt(f'{wdir}/calib2struct.mat',
        f'{wdir}/M0_AP_vol0.nii.gz',
        f'{wdir}/T1_biascorr_brain.nii.gz'
    )

    a2s = rt.chain(a2c, c2s)

    wdir = str(REAL_ROOT / 'common')
    cpath = f'{wdir}/sub-SS_run-01_scale-2_space-common_M0.nii.gz'

    tivol_cmn = a2s.apply_to_array(tivol, spc, cpath)
    rt.ImageSpace.save_like(cpath, tivol_cmn, f'{wdir}/tivol.nii.gz')

    # subject-02 repeats 
    wdir = str(REAL_ROOT / '../maastricht-02/reg')
    for run in range(3):
        a2c = rt.Registration.from_flirt(f'{wdir}/run{run+1}_to_calib.mat',
            f'{wdir}/sub-02_run-1_asl_vol0.nii.gz', 
            f'{wdir}/M0_AP_vol0.nii.gz'
        )

        c2s = rt.Registration.from_flirt(f'{wdir}/calib2struct.mat',
            f'{wdir}/M0_AP_vol0.nii.gz',
            f'{wdir}/T1_biascorr_brain.nii.gz'
        )

        a2s = rt.chain(a2c, c2s)
        nat_spc = rt.ImageSpace(f'{wdir}/../native/sub-02_run-01_scale-2_space-native_M0.nii.gz')
        cmn_spc = rt.ImageSpace(f'{wdir}/../common/sub-02_run-01_scale-2_space-common_M0.nii.gz')
        tivol = np.indices(nat_spc.size, dtype=np.float32)[-1]
        tivol *= (2 * 0.0325)
        tivol += 1.8
        tivol = np.stack(100 * [tivol], axis=-1)

        tivol_path = f'{wdir}/../native/tivol.nii.gz'
        nat_spc.save_image(tivol, tivol_path)

        tivol_cmn = a2s.apply_to_array(tivol, nat_spc, cmn_spc)
        tivol_path = f'{wdir}/../common/tivol_rpt{run+1}.nii.gz'
        cmn_spc.save_image(tivol_cmn, tivol_path)


def fit_real(): 

    jobs = []

    # Native space 
    wdir = str(REAL_ROOT / 'native')
    apath = f'{wdir}/sub-SS_run-01_scale-2_space-native_asl.nii.gz'
    mpath = f'{wdir}/sub-SS_run-01_scale-2_space-native_mask.nii.gz'
    cpath = f'{wdir}/sub-SS_run-01_scale-2_space-native_M0.nii.gz'
    pvs = [f'{wdir}/sub-SS_run-01_scale-2_space-native_pvgm_tob.nii.gz',
          f'{wdir}/sub-SS_run-01_scale-2_space-native_pvwm_tob.nii.gz']
    tipath = f'{wdir}/tivol.nii.gz'

    # Native space, no resampling. 
    bopts = { 'bat': 1.3, 'batwm': 1.6, 'fixbat': True, 'tiimg': tipath }
    odir = f'{wdir}/basil_native'
    jobs.append(
        fabber_funcs.oxasl_cmd(apath, cpath, mpath, odir, bopts, pvs)
    )

    odir = f'{wdir}/lr_native'
    jobs += lr_method.oxasl_lr(apath, cpath, mpath, odir, bopts, pvs)

    # Native space, resampled data, un-resampled pvs 
    apath = f'{wdir}/asl_resamp.nii.gz'
    odir = f'{wdir}/basil_resampled'
    jobs.append(
        fabber_funcs.oxasl_cmd(apath, cpath, mpath, odir, bopts, pvs)
    )

    odir = f'{wdir}/lr_resampled'
    jobs += lr_method.oxasl_lr(apath, cpath, mpath, odir, bopts, pvs)

    # Native space, resampled data, resampled PVs 
    pvs = [f'{wdir}/pvgm_resamp.nii.gz',
          f'{wdir}/pvwm_resamp.nii.gz']
    cpath = f'{wdir}/m0_resamp.nii.gz'
    odir = f'{wdir}/basil_double'
    jobs.append(
        fabber_funcs.oxasl_cmd(apath, cpath, mpath, odir, bopts, pvs)
    )

    odir = f'{wdir}/lr_double'
    jobs += lr_method.oxasl_lr(apath, cpath, mpath, odir, bopts, pvs)

    # Common space
    wdir = str(REAL_ROOT / 'common')
    apath = f'{wdir}/sub-SS_run-01_scale-2_space-common_asl.nii.gz'
    mpath = f'{wdir}/sub-SS_run-01_scale-2_space-common_mask.nii.gz'
    cpath = f'{wdir}/sub-SS_run-01_scale-2_space-common_M0.nii.gz'
    pvs = [f'{wdir}/sub-SS_run-01_scale-2_space-common_pvgm_tob.nii.gz',
          f'{wdir}/sub-SS_run-01_scale-2_space-common_pvwm_tob.nii.gz']
    tipath = f'{wdir}/tivol.nii.gz'

    # Common space, resampled data with double-resampled PVs
    odir = f'{wdir}/basil_double'
    jobs.append(
        fabber_funcs.oxasl_cmd(apath, cpath, mpath, odir, bopts, pvs)
    )

    odir = f'{wdir}/lr_double'
    jobs += lr_method.oxasl_lr(apath, cpath, mpath, odir, bopts, pvs)

    # Common space, resampled data with naive PVs 
    pvs = [f'{wdir}/sub-SS_scale-2_space-common_pvgm_tob_naive.nii.gz',
           f'{wdir}/sub-SS_scale-2_space-common_pvwm_tob_naive.nii.gz']

    odir = f'{wdir}/basil_naive'
    jobs.append(
        fabber_funcs.oxasl_cmd(apath, cpath, mpath, odir, bopts, pvs)
    )

    odir = f'{wdir}/lr_naive'
    jobs += lr_method.oxasl_lr(apath, cpath, mpath, odir, bopts, pvs)

    for rpt in range(3):

        # Repeat acquisitions, native space 
        wdir = str(REAL_ROOT / '../maastricht-02/native')
        apath = f'{wdir}/sub-02_run-{rpt+1:02d}_scale-2_space-native_asl.nii.gz'
        mpath = f'{wdir}/sub-02_run-{rpt+1:02d}_scale-2_space-native_mask.nii.gz'
        cpath = f'{wdir}/sub-02_run-{rpt+1:02d}_scale-2_space-native_M0.nii.gz'
        pvs = [f'{wdir}/sub-02_run-{rpt+1:02d}_scale-2_space-native_pvgm_tob.nii.gz',
          f'{wdir}/sub-02_run-{rpt+1:02d}_scale-2_space-native_pvwm_tob.nii.gz']
        tipath = f'{wdir}/tivol.nii.gz'

        bopts = { 'bat': 1.3, 'batwm': 1.6, 'fixbat': True, 'tiimg': tipath }
        odir = f'{wdir}/basil_native_rpt{rpt+1}'
        jobs.append(
            fabber_funcs.oxasl_cmd(apath, cpath, mpath, odir, bopts, pvs)
        )

        odir = f'{wdir}/lr_native_rpt{rpt+1}'
        jobs += lr_method.oxasl_lr(apath, cpath, mpath, odir, bopts, pvs)

        # Common space 
        wdir = str(REAL_ROOT / '../maastricht-02/common')
        apath = f'{wdir}/sub-02_run-{rpt+1:02d}_scale-2_space-common_asl.nii.gz'
        mpath = f'{wdir}/sub-02_run-{rpt+1:02d}_scale-2_space-common_mask.nii.gz'
        cpath = f'{wdir}/sub-02_run-{rpt+1:02d}_scale-2_space-common_M0.nii.gz'
        pvs = [f'{wdir}/sub-02_run-{rpt+1:02d}_scale-2_space-common_pvgm_tob.nii.gz',
          f'{wdir}/sub-02_run-{rpt+1:02d}_scale-2_space-common_pvwm_tob.nii.gz']
        tipath = f'{wdir}/tivol.nii.gz'

        # Double resampled pvs 
        bopts = { 'bat': 1.3, 'batwm': 1.6, 'fixbat': True, 'tiimg': tipath }
        odir = f'{wdir}/basil_double_rpt{rpt+1}'
        jobs.append(
            fabber_funcs.oxasl_cmd(apath, cpath, mpath, odir, bopts, pvs)
        )

        odir = f'{wdir}/lr_double_rpt{rpt+1}'
        jobs += lr_method.oxasl_lr(apath, cpath, mpath, odir, bopts, pvs)

        # Naive PVs 
        pvs = [f'{wdir}/sub-02_scale-2_space-common_pvgm_tob_naive.nii.gz',
          f'{wdir}/sub-02_scale-2_space-common_pvwm_tob_naive.nii.gz']

        bopts = { 'bat': 1.3, 'batwm': 1.6, 'fixbat': True, 'tiimg': tipath }
        odir = f'{wdir}/basil_naive_rpt{rpt+1}'
        jobs.append(
            fabber_funcs.oxasl_cmd(apath, cpath, mpath, odir, bopts, pvs)
        )

        odir = f'{wdir}/lr_naive_rpt{rpt+1}'
        jobs += lr_method.oxasl_lr(apath, cpath, mpath, odir, bopts, pvs)

    worker = functools.partial(subprocess.run, shell=True)
    with multiprocessing.Pool(2) as p: 
        p.map(worker, jobs)



if __name__ == '__main__':

    prepare_sim_data()
    fit_simulations()

    # prepare_real_data()
    # fit_real() 