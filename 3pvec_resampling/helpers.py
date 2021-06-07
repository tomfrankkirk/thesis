import nibabel
import os.path as op 
import numpy as np 
import toblerone

def load_basil(from_dir, pvec=True):
    if pvec: 
        gm = op.join(from_dir, 'basil_out', 'step2', 'mean_ftiss.nii.gz')
        wm = op.join(from_dir, 'basil_out', 'step2', 'mean_fwm.nii.gz')
        return np.stack([
            nibabel.load(p).get_fdata() for p in [gm, wm]
        ], axis=-1)
    
    else: 
        gm = op.join(from_dir, 'basil_out', 'step1', 'mean_ftiss.nii.gz')
        return nibabel.load(gm).get_fdata()
    
def load_basil_att(from_dir, pvec=True):
    if pvec: 
        gm = op.join(from_dir, 'basil_out', 'step2', 'mean_delttiss.nii.gz')
        wm = op.join(from_dir, 'basil_out', 'step2', 'mean_deltwm.nii.gz')
        return np.stack([
            nibabel.load(p).get_fdata() for p in [gm, wm]
        ], axis=-1)
    
    else: 
        gm = op.join(from_dir, 'basil_out', 'step1', 'mean_delttiss.nii.gz')
        return nibabel.load(gm).get_fdata()
    
def load_oxasl(from_dir, pvec=True):
    if pvec: 
        gm = op.join(from_dir, 'output_pvcorr', 'native', 'perfusion_calib.nii.gz')
        wm = op.join(from_dir, 'output_pvcorr', 'native', 'perfusion_wm_calib.nii.gz')
        return np.stack([
            nibabel.load(p).get_fdata() for p in [gm, wm]
        ], axis=-1)
    
    else: 
        gm = op.join(from_dir, 'output', 'native', 'perfusion.nii.gz')
        return nibabel.load(gm).get_fdata()
    
def load_lr(from_dir):
    gm = op.join(from_dir, 'gm', 'basil_out', 'step1', 'mean_ftiss.nii.gz')
    wm = op.join(from_dir, 'wm', 'basil_out', 'step1', 'mean_ftiss.nii.gz')
    return np.stack([
        nibabel.load(p).get_fdata() for p in [gm, wm]
    ], axis=-1)

def load_oxasl_lr(from_dir, pvec=True):
    gm = op.join(from_dir, 'gm', 'output', 'native', 'perfusion_calib.nii.gz')
    wm = op.join(from_dir, 'wm', 'output', 'native', 'perfusion_calib.nii.gz')
    return np.stack([
        nibabel.load(p).get_fdata() for p in [gm, wm]
    ], axis=-1)


def load_svb_hybrid(from_dir):
    
    ftiss = np.load(op.join(from_dir, 'mean_ftiss.npz'))['arr_0']
    att = np.load(op.join(from_dir, 'mean_delttiss.npz'))['arr_0']
    proj = toblerone.Projector.load(op.join(from_dir, 'projector.h5'))
    
    n_nodes = ftiss.size 
    n_surf = proj.n_surf_nodes
    n_subcort = proj.n_subcortical_nodes
    n_vox = n_nodes - n_surf - n_subcort 
    
    cbf = [
        ftiss[:n_surf], 
        ftiss[n_surf:n_nodes - n_subcort],
        ftiss[-n_subcort:], 
    ]
    
    att = [
        att[:proj.n_surf_nodes], 
        att[proj.n_surf_nodes:proj.n_nodes - proj.n_subcortical_nodes],
        att[proj.n_nodes - proj.n_subcortical_nodes:], 
    ]
    
    return cbf, att, proj 

def load_svb_ak(from_dir): 

    ftiss = np.load(op.join(from_dir, 'ftiss_ak_history.txt'))

    
def load_basil_ak(from_dir, idx):
    log = open(op.join(from_dir, 'basil_out/step2/logfile')).read()
    idx = log.index('*** Spatial iteration *** 200')
    log = log[idx:]
    idx = log.index(f'SpatialPrior::Calculate aK {idx}: New aK: ')
    log = log[idx + len(f'SpatialPrior::Calculate aK {idx}: New aK: '):]
    idx = log.index('\n')
    return float(log[:idx])

def load_basil_projected(from_dir):
    g1 = op.join(from_dir, 'L_cortex_ftiss.func.gii')
    g2 = op.join(from_dir, 'R_cortex_ftiss.func.gii')
    return np.concatenate([
        nibabel.load(g).darrays[0].data
        for g in [g1, g2]
    ])